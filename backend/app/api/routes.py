"""HTTP surface.

Handlers stay thin on purpose: validate, delegate to a service, serialise.
All domain logic lives under ``app/core`` and ``app/services`` so it can be
exercised without an HTTP client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.core.llm import LLMUnavailableError, check_ollama
from app.core.paths import (
    PathSecurityError,
    normalize_root,
    resolve_within,
    unique_destination,
)
from app.schemas import (
    ChatRequest,
    FilingRequest,
    HealthResponse,
    IndexRequest,
    JobResponse,
    SearchRequest,
    WatchRequest,
)
from app.services.container import Container, get_container

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

ContainerDep = Annotated[Container, Depends(get_container)]

# asyncio only holds a weak reference to a running task, so a fire-and-forget
# task can be garbage-collected mid-flight. Keeping a strong reference until it
# completes is the documented fix.
job_tasks: set[asyncio.Task[None]] = set()


def _root_or_400(raw: str, container: Container) -> Path:
    try:
        return normalize_root(raw, container.settings)
    except PathSecurityError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


def _require_index(root: Path, container: Container) -> None:
    if container.repository.count(root) == 0:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"'{root}' has not been indexed yet. Run POST /api/index first.",
        )


# --------------------------------------------------------------------- health
@router.get("/health", response_model=HealthResponse)
async def health(container: ContainerDep) -> HealthResponse:
    ollama = await check_ollama(container.settings)
    healthy = ollama.reachable and ollama.chat_model_ready and ollama.embedding_model_ready
    return HealthResponse(
        status="ok" if healthy else "degraded",
        ollama={
            "reachable": ollama.reachable,
            "baseUrl": ollama.base_url,
            "models": list(ollama.models),
            "chatModelReady": ollama.chat_model_ready,
            "embeddingModelReady": ollama.embedding_model_ready,
            "detail": ollama.detail,
        },
        indexedRoots=len(container.manifests.list_roots()),
        chatModel=container.settings.chat_model,
        embeddingModel=container.settings.embedding_model,
        dataDir=str(container.settings.data_dir),
    )


# -------------------------------------------------------------------- indexing
@router.post("/index", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_index(request: IndexRequest, container: ContainerDep) -> JobResponse:
    """Kick off an index run and return immediately with a job handle."""
    root = _root_or_400(request.path, container)
    job = container.jobs.create(kind="index", target=str(root))

    task = asyncio.create_task(
        container.jobs.run(
            job,
            container.indexer.index_root,
            root,
            force=request.force,
            summarize=request.summarize,
        )
    )
    job_tasks.add(task)
    task.add_done_callback(job_tasks.discard)

    return JobResponse(**job.to_dict())


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def job_status(job_id: str, container: ContainerDep) -> JobResponse:
    job = container.jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No such job: {job_id}")
    return JobResponse(**job.to_dict())


@router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: str, container: ContainerDep) -> JobResponse:
    if not container.jobs.cancel(job_id):
        raise HTTPException(status.HTTP_409_CONFLICT, "Job is not cancellable.")
    job = container.jobs.get(job_id)
    assert job is not None
    return JobResponse(**job.to_dict())


# ----------------------------------------------------------------- watching
@router.post("/watch")
async def set_watch(request: WatchRequest, container: ContainerDep) -> dict[str, Any]:
    """Turn live re-indexing on or off for a root.

    Enabling starts a filesystem watcher that debounces changes and runs the
    same incremental ``index_root`` a manual "Index" click uses, so a file
    dropped into (or edited within) the folder becomes searchable within a
    few seconds of the last write, with no further action needed.
    """
    root = _root_or_400(request.path, container)
    if request.enabled:
        return container.watchers.start(root)
    # stop() joins the observer's OS thread; keep that off the event loop.
    await asyncio.to_thread(container.watchers.stop, root)
    return {"root": str(root), "watching": False}


@router.get("/watch")
async def watch_status(
    container: ContainerDep,
    path: Annotated[str, Query(..., min_length=1)],
) -> dict[str, Any]:
    root = _root_or_400(path, container)
    return container.watchers.status(root) or {"root": str(root), "watching": False}


@router.get("/roots")
async def list_roots(container: ContainerDep) -> dict[str, Any]:
    """Previously indexed folders, newest first."""
    return {
        "roots": [
            {
                "path": manifest.root,
                "updatedAt": manifest.updated_at,
                "files": manifest.stats.get("total_files", len(manifest.files)),
                "chunks": manifest.stats.get("chunks", 0),
                "totalSize": manifest.stats.get("total_size", 0),
                "exists": Path(manifest.root).is_dir(),
            }
            for manifest in container.manifests.list_roots()
        ]
    }


@router.delete("/roots")
async def forget_root(
    container: ContainerDep,
    path: Annotated[str, Query(..., min_length=1)],
) -> dict[str, str]:
    root = _root_or_400(path, container)
    # A watcher re-indexing a collection that is about to be dropped would
    # immediately recreate it out from under the delete.
    await asyncio.to_thread(container.watchers.stop, root)
    container.repository.drop(root)
    container.manifests.delete(root)
    return {"detail": f"Index for {root} removed."}


# ------------------------------------------------------------------ directory
@router.get("/tree")
async def directory_tree(
    container: ContainerDep,
    path: Annotated[str, Query(..., min_length=1)],
    max_depth: Annotated[int, Query(ge=1, le=12)] = 4,
    include_files: bool = True,
) -> dict[str, Any]:
    """Live directory tree, decorated with any cached folder summaries.

    Read from disk rather than the index so the view is never stale, then
    joined against stored summaries by relative path.
    """
    root = _root_or_400(path, container)
    scan = await asyncio.to_thread(container.scanner.scan, root)
    summaries = container.manifests.load(root).folder_summaries

    def decorate(node: Any) -> None:
        node.summary = summaries.get(node.rel_path)
        for child in node.children:
            decorate(child)

    decorate(scan.tree)

    def prune(node: Any, depth: int) -> dict[str, Any]:
        payload = {
            "name": node.name,
            "relPath": node.rel_path,
            "absPath": node.abs_path,
            "depth": node.depth,
            "directFiles": len(node.files),
            "totalFiles": node.total_files,
            "totalSize": node.total_size,
            "summary": node.summary,
            "truncated": depth >= max_depth and bool(node.children),
            "children": (
                [prune(child, depth + 1) for child in node.children]
                if depth < max_depth
                else []
            ),
            "files": (
                [
                    {
                        "name": f.name,
                        "relPath": f.rel_path,
                        "ext": f.ext,
                        "size": f.size_bytes,
                        "modifiedAt": f.modified_at,
                    }
                    for f in sorted(node.files, key=lambda f: f.name.lower())[:40]
                ]
                if include_files
                else []
            ),
        }
        return payload

    return {
        "root": str(root),
        "totalFiles": scan.tree.total_files,
        "totalSize": scan.tree.total_size,
        "directories": scan.directories_scanned,
        "indexedChunks": container.repository.count(root),
        "truncated": scan.truncated,
        "tree": prune(scan.tree, 0),
    }


# --------------------------------------------------------------------- search
@router.post("/search")
async def search(request: SearchRequest, container: ContainerDep) -> dict[str, Any]:
    """Retrieval only -- no generation. Fast, and useful for debugging recall."""
    root = _root_or_400(request.path, container)
    _require_index(root, container)
    try:
        result = await container.rag.search(root, request.query, request.top_k)
    except LLMUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return {
        "plan": result.plan.to_dict(),
        "sources": result.sources_payload(),
        "tookMs": round(result.took_ms, 1),
    }


def _sse(event: str, payload: dict[str, Any]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# How often to emit an SSE comment while waiting on the next Ollama token.
# A CPU-bound 8B model can go quiet for a while between tokens; without a
# heartbeat, some proxies/antivirus/VPN stacks treat that gap as a dead
# connection and reset it, which surfaces in the browser as a bare
# "network error" thrown out of the fetch body reader -- not as our own
# SSE "error" event, since from the server's point of view nothing failed.
_CHAT_HEARTBEAT_SECONDS = 12.0


@router.post("/chat")
async def chat(request: ChatRequest, container: ContainerDep) -> Any:
    """Grounded answer over the indexed folder.

    Streams as SSE: a ``sources`` frame first so the UI can render citations
    while the model is still writing, then ``token`` frames, then ``done``.
    """
    root = _root_or_400(request.path, container)
    _require_index(root, container)

    print(f"[chat] request root={root} question={request.question!r}")

    try:
        result = await container.rag.search(root, request.question, request.top_k)
    except LLMUnavailableError as exc:
        print(f"[chat] retrieval failed, Ollama unavailable: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - never let retrieval crash the request
        logger.exception("search failed for %s", root)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Search failed: {exc}") from exc

    print(
        f"[chat] retrieval ok in {result.took_ms:.0f}ms, {len(result.chunks)} chunks -> "
        f"generating with model={container.settings.chat_model!r} "
        f"base_url={container.settings.ollama_base_url!r}"
    )

    if not request.stream:
        chunks = [token async for token in container.rag.answer_stream(request.question, result)]
        return {
            "answer": "".join(chunks),
            "plan": result.plan.to_dict(),
            "sources": result.sources_payload(),
        }

    async def event_stream() -> AsyncIterator[str]:
        yield _sse("plan", result.plan.to_dict())
        yield _sse("sources", {"sources": result.sources_payload()})

        # Run generation in a background task and hand tokens back through a
        # queue, so this loop is free to emit a heartbeat comment whenever
        # the queue goes quiet for too long instead of blocking on the model.
        queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()

        async def produce() -> None:
            token_count = 0
            gen_started = time.perf_counter()
            try:
                async for token in container.rag.answer_stream(request.question, result):
                    token_count += 1
                    await queue.put(("token", token))
            except asyncio.CancelledError:
                print("[chat] generation cancelled (client disconnected)")
                raise
            except Exception as exc:  # noqa: BLE001 - must not break the SSE frame
                # This is the block to watch in your terminal: it fires for a
                # dead Ollama connection, a missing model, a request timeout,
                # or any other failure *inside* generation.
                logger.exception(
                    "chat generation failed root=%s model=%s base_url=%s after %d token(s)",
                    root,
                    container.settings.chat_model,
                    container.settings.ollama_base_url,
                    token_count,
                )
                print(f"[chat] GENERATION ERROR: {type(exc).__name__}: {exc}")
                await queue.put(("error", f"{type(exc).__name__}: {exc}"))
            else:
                elapsed = time.perf_counter() - gen_started
                print(f"[chat] generation ok: {token_count} tokens in {elapsed:.1f}s")
            finally:
                await queue.put(None)

        producer = asyncio.create_task(produce())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=_CHAT_HEARTBEAT_SECONDS)
                except TimeoutError:
                    # Comment lines (leading ':') are valid, inert SSE frames.
                    # They reset any idle-connection timer downstream without
                    # the frontend having to parse them as real events.
                    yield ": keep-alive\n\n"
                    continue
                if item is None:
                    break
                kind, payload = item
                if kind == "token":
                    yield _sse("token", {"t": payload})
                else:
                    yield _sse("error", {"detail": payload})
        except asyncio.CancelledError:
            producer.cancel()
            raise
        yield _sse("done", {"tookMs": round(result.took_ms, 1)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Tells nginx (and Next's dev proxy) not to buffer, which would
            # defeat streaming entirely.
            "X-Accel-Buffering": "no",
        },
    )


# -------------------------------------------------------------------- cleanup
@router.get("/cleanup")
async def cleanup(
    container: ContainerDep,
    path: Annotated[str, Query(..., min_length=1)],
    narrate: bool = True,
) -> dict[str, Any]:
    root = _root_or_400(path, container)
    scan = await asyncio.to_thread(container.scanner.scan, root)
    report = await asyncio.to_thread(container.cleanup.analyze, root, scan.files)
    if narrate:
        report.narrative = await container.cleanup.narrate(report)
    return report.to_dict()


# --------------------------------------------------------------------- filing
@router.post("/filing/suggest")
async def filing_suggest(request: FilingRequest, container: ContainerDep) -> dict[str, Any]:
    """Propose (and optionally perform) a destination for one file."""
    root = _root_or_400(request.path, container)
    _require_index(root, container)

    source = Path(request.file).expanduser()
    if not source.is_absolute():
        source = root / request.file
    source = source.resolve()
    if not source.is_file():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Not a file: {source}")

    proposal = await asyncio.to_thread(
        container.filing.run, root, source, not request.apply, request.apply
    )
    if proposal.error and not proposal.destination_dir:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, proposal.error)
    return proposal.to_dict()


@router.post("/filing/apply")
async def filing_apply(request: FilingRequest, container: ContainerDep) -> dict[str, Any]:
    """Execute a move the user has explicitly confirmed.

    Separate from ``/filing/suggest`` so the destructive path always requires
    a distinct, deliberate call.
    """
    root = _root_or_400(request.path, container)
    source = Path(request.file).expanduser()
    if not source.is_absolute():
        source = root / request.file
    source = source.resolve()
    if not source.is_file():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Not a file: {source}")
    if not request.destination:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "destination is required.")

    try:
        destination_dir = resolve_within(root, request.destination)
    except PathSecurityError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        target = unique_destination(destination_dir, request.filename or source.name)
        await asyncio.to_thread(shutil.move, str(source), str(target))
    except (OSError, shutil.Error) as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Move failed: {exc}") from exc

    return {
        "applied": True,
        "sourcePath": str(source),
        "finalPath": str(target),
        "destinationDir": str(destination_dir),
    }
