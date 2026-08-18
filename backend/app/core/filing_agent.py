"""Auto-filing agent built on LangGraph.

Graph
-----
    inspect -> retrieve -> rank -> decide -> [route] -> apply -> END
                                                     \\-> END (proposal only)

Why a graph rather than a straight function? The steps have genuinely
different failure modes and one real branch (apply vs. propose). Modelling it
as a ``StateGraph`` gives each step an isolated, inspectable input/output, lets
the conditional edge express the auto-apply policy declaratively, and means
adding a human-approval node later is a new node plus one edge.

Why not a tool-calling ReAct agent? Because the action space is one move
operation and the search space is a ranked candidate list. A fixed graph is
deterministic, needs exactly one LLM call, and cannot loop forever -- all of
which matter when the side effect is moving somebody's files.

The critical safety property is in :meth:`_decide`: the LLM chooses *by index*
from a list we constructed, and the choice is validated against that list. A
hallucinated path is impossible to act on, because a hallucinated path is not
in the candidate set and we fall back to the top-ranked destination.
"""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.core.extractors import ExtractorRegistry
from app.core.llm import LLMUnavailableError, complete, embed_query
from app.core.paths import (
    PathSecurityError,
    human_size,
    is_within,
    resolve_within,
    unique_destination,
)
from app.core.textutils import extract_json, truncate
from app.core.vectorstore import KIND_FOLDER_SUMMARY, VectorRepository

logger = logging.getLogger(__name__)

_DECIDE_SYSTEM = """You file incoming documents into an existing folder structure.

You will see the document and a numbered list of candidate destination folders.
Choose the folder where a careful archivist would put this document, based on
what already lives in each folder.

Return ONLY this JSON object:
{
  "choice": <integer, the number of a candidate from the list>,
  "confidence": <float between 0 and 1>,
  "reason": "<one sentence, referencing what the folder already contains>",
  "suggested_filename": "<a clearer filename, or \\"\\" to keep the current one>"
}

Rules:
- "choice" MUST be one of the listed numbers. Never invent a path.
- Use a confidence below 0.5 if no candidate is a good home.
- Keep the original file extension in suggested_filename."""


@dataclass(slots=True)
class Candidate:
    rel_path: str
    abs_path: str
    semantic_score: float = 0.0
    hits: int = 0
    ext_affinity: float = 0.0
    summary: str = ""
    example_files: list[str] = field(default_factory=list)
    final_score: float = 0.0


class FilingState(TypedDict, total=False):
    """Shared state threaded through the graph.

    LangGraph merges each node's returned dict into this object, so nodes stay
    pure functions of ``state`` and never mutate shared globals. Keys are left
    un-annotated (no reducer), which gives last-write-wins semantics -- correct
    here because the graph is linear and no two nodes write the same key.
    """

    root: str
    source_path: str
    dry_run: bool
    auto_apply: bool

    file_name: str
    file_ext: str
    file_size: int
    content_preview: str

    neighbours: list[dict[str, Any]]
    candidates: list[dict[str, Any]]

    destination: str
    destination_rel: str
    suggested_filename: str
    confidence: float
    reason: str
    applied: bool
    final_path: str
    error: str


@dataclass(slots=True)
class FilingProposal:
    source_path: str
    destination_dir: str
    destination_rel: str
    suggested_filename: str
    confidence: float
    reason: str
    applied: bool
    final_path: str
    candidates: list[dict[str, Any]]
    neighbours: list[dict[str, Any]]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourcePath": self.source_path,
            "destinationDir": self.destination_dir,
            "destinationRel": self.destination_rel,
            "suggestedFilename": self.suggested_filename,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "applied": self.applied,
            "finalPath": self.final_path,
            "candidates": self.candidates,
            "neighbours": self.neighbours,
            "error": self.error,
        }


class FilingAgent:
    def __init__(
        self,
        settings: Settings,
        repository: VectorRepository,
        registry: ExtractorRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._registry = registry or ExtractorRegistry()
        self._graph = self._build_graph()

    # ------------------------------------------------------------- the graph
    def _build_graph(self):
        builder = StateGraph(FilingState)
        builder.add_node("inspect", self._inspect)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("rank", self._rank)
        builder.add_node("decide", self._decide)
        builder.add_node("apply", self._apply)

        builder.add_edge(START, "inspect")
        builder.add_edge("inspect", "retrieve")
        builder.add_edge("retrieve", "rank")
        builder.add_edge("rank", "decide")
        builder.add_conditional_edges(
            "decide",
            self._should_apply,
            {"apply": "apply", "propose": END},
        )
        builder.add_edge("apply", END)
        return builder.compile()

    def run(
        self,
        root: Path,
        source: Path,
        dry_run: bool = True,
        auto_apply: bool = False,
    ) -> FilingProposal:
        state: FilingState = self._graph.invoke(
            {
                "root": str(root),
                "source_path": str(source),
                "dry_run": dry_run,
                "auto_apply": auto_apply,
            }
        )
        return FilingProposal(
            source_path=state.get("source_path", str(source)),
            destination_dir=state.get("destination", ""),
            destination_rel=state.get("destination_rel", ""),
            suggested_filename=state.get("suggested_filename", source.name),
            confidence=float(state.get("confidence", 0.0)),
            reason=state.get("reason", ""),
            applied=bool(state.get("applied", False)),
            final_path=state.get("final_path", ""),
            candidates=state.get("candidates", []),
            neighbours=state.get("neighbours", []),
            error=state.get("error", ""),
        )

    # ----------------------------------------------------------------- nodes
    def _inspect(self, state: FilingState) -> FilingState:
        """Read the incoming file's metadata and a bounded content preview."""
        source = Path(state["source_path"])
        if not source.is_file():
            return {"error": f"Not a file: {source}"}

        text, _kind = self._registry.extract(
            source, self._settings.filing_content_preview_chars
        )
        return {
            "file_name": source.name,
            "file_ext": source.suffix.lower(),
            "file_size": source.stat().st_size,
            "content_preview": text,
        }

    def _retrieve(self, state: FilingState) -> FilingState:
        """Find existing documents that resemble the incoming one.

        The insight: the right home for a file is wherever its nearest
        semantic neighbours already live. Filename plus content preview is
        embedded and matched against the index, so we are effectively asking
        "what does this look like, and where does that kind of thing go?".
        """
        if state.get("error"):
            return {}

        probe = f"File: {state.get('file_name', '')}\n\n{state.get('content_preview', '')}"
        try:
            embedding = embed_query(truncate(probe, 4000, ""))
        except LLMUnavailableError as exc:
            return {"error": f"Embedding unavailable: {exc}"}

        root = Path(state["root"])
        chunks = self._repository.query(root, embedding, self._settings.retrieval_fetch_k)
        source_path = str(Path(state["source_path"]).resolve())

        neighbours = [
            {
                "relPath": chunk.rel_path,
                "parentRel": str(chunk.metadata.get("parent_rel", "")),
                "name": str(chunk.metadata.get("name", "")),
                "ext": str(chunk.metadata.get("ext", "")),
                "kind": str(chunk.metadata.get("kind", "")),
                "score": round(chunk.score, 4),
                "excerpt": truncate(chunk.document, 240, "..."),
            }
            for chunk in chunks
            # Never let the file recommend its own current location as
            # evidence -- it may already be indexed.
            if chunk.abs_path != source_path
        ]
        return {"neighbours": neighbours}

    def _rank(self, state: FilingState) -> FilingState:
        """Aggregate neighbours into scored destination folders.

        score = 0.60 * best semantic similarity in the folder
              + 0.25 * evidence weight (how many neighbours point here)
              + 0.15 * extension affinity (does this folder already hold .pdf?)

        The first term captures topical fit; the second stops a single lucky
        chunk from outvoting a folder with five solid matches; the third
        encodes the very real convention that people group by file type.
        Weights are explicit and tunable rather than buried in a prompt.
        """
        if state.get("error"):
            return {}

        root = Path(state["root"])
        summaries = self._repository.get_folder_summaries(root)
        by_folder: dict[str, Candidate] = {}
        ext = state.get("file_ext", "")
        ext_counts: Counter[str] = Counter()

        for neighbour in state.get("neighbours", []):
            if neighbour["kind"] == KIND_FOLDER_SUMMARY:
                folder_rel = neighbour["relPath"]
            else:
                folder_rel = neighbour["parentRel"]

            candidate = by_folder.get(folder_rel)
            if candidate is None:
                candidate = Candidate(
                    rel_path=folder_rel,
                    abs_path=str(root / folder_rel) if folder_rel else str(root),
                    summary=summaries.get(folder_rel, ""),
                )
                by_folder[folder_rel] = candidate

            candidate.semantic_score = max(candidate.semantic_score, neighbour["score"])
            candidate.hits += 1
            if neighbour["name"] and len(candidate.example_files) < 5:
                candidate.example_files.append(neighbour["name"])
            if neighbour["ext"] == ext:
                ext_counts[folder_rel] += 1

        if not by_folder:
            # Cold index: fall back to the root so the agent still returns a
            # usable (low-confidence) proposal instead of failing.
            by_folder[""] = Candidate(
                rel_path="", abs_path=str(root), summary=summaries.get("", "")
            )

        max_hits = max((c.hits for c in by_folder.values()), default=1) or 1
        for rel, candidate in by_folder.items():
            candidate.ext_affinity = min(1.0, ext_counts[rel] / 3.0)
            candidate.final_score = (
                0.60 * candidate.semantic_score
                + 0.25 * (candidate.hits / max_hits)
                + 0.15 * candidate.ext_affinity
            )

        ranked = sorted(by_folder.values(), key=lambda c: -c.final_score)
        top = ranked[: self._settings.filing_candidate_dirs]
        return {
            "candidates": [
                {
                    "relPath": c.rel_path,
                    "absPath": c.abs_path,
                    "score": round(c.final_score, 4),
                    "semanticScore": round(c.semantic_score, 4),
                    "matches": c.hits,
                    "summary": c.summary,
                    "exampleFiles": c.example_files,
                }
                for c in top
            ]
        }

    def _decide(self, state: FilingState) -> FilingState:
        """Ask the LLM to pick a candidate, then validate the answer.

        Falls back to the top-ranked candidate whenever the model is
        unavailable or returns something outside the allowed index range.
        """
        if state.get("error"):
            return {}

        candidates = state.get("candidates", [])
        if not candidates:
            return {"error": "No candidate destinations. Index the folder first."}

        top = candidates[0]
        fallback: FilingState = {
            "destination": top["absPath"],
            "destination_rel": top["relPath"],
            "suggested_filename": state.get("file_name", ""),
            "confidence": float(top["score"]),
            "reason": (
                f"Closest semantic match ({top['matches']} similar documents already "
                f"in {top['relPath'] or 'the root folder'})."
            ),
        }

        listing = "\n".join(
            f"{position}. {c['relPath'] or '(root folder)'}\n"
            f"   summary: {c['summary'] or 'n/a'}\n"
            f"   already contains: {', '.join(c['exampleFiles']) or 'n/a'}\n"
            f"   match score: {c['score']:.2f} from {c['matches']} similar chunks"
            for position, c in enumerate(candidates, start=1)
        )
        prompt = (
            f"Incoming document\n"
            f"  name: {state.get('file_name')}\n"
            f"  type: {state.get('file_ext') or 'none'}\n"
            f"  size: {human_size(state.get('file_size', 0))}\n"
            f"  content preview:\n{truncate(state.get('content_preview', ''), 1800, '...')}\n\n"
            f"Candidate destinations:\n{listing}\n\n"
            f"Respond with the JSON object."
        )

        try:
            raw = complete(_DECIDE_SYSTEM, prompt)
        except LLMUnavailableError as exc:
            logger.info("filing decision fell back to ranking: %s", exc)
            fallback["reason"] += " (LLM unavailable; ranked heuristically.)"
            return fallback

        payload = extract_json(raw, dict)
        if not payload:
            return fallback

        try:
            choice = int(payload.get("choice", 0))
        except (TypeError, ValueError):
            return fallback
        if not 1 <= choice <= len(candidates):
            logger.warning("LLM chose out-of-range candidate %s", choice)
            return fallback

        chosen = candidates[choice - 1]
        filename = str(payload.get("suggested_filename") or "").strip()
        return {
            "destination": chosen["absPath"],
            "destination_rel": chosen["relPath"],
            "suggested_filename": self._safe_filename(
                filename, state.get("file_name", ""), state.get("file_ext", "")
            ),
            "confidence": max(0.0, min(1.0, float(payload.get("confidence", 0.5)))),
            "reason": str(payload.get("reason") or "").strip() or fallback["reason"],
        }

    def _should_apply(self, state: FilingState) -> str:
        """Conditional edge: only move files when explicitly authorised."""
        if state.get("error"):
            return "propose"
        if state.get("dry_run", True) or not state.get("auto_apply", False):
            return "propose"
        if state.get("confidence", 0.0) < self._settings.filing_confidence_threshold:
            return "propose"
        return "apply"

    def _apply(self, state: FilingState) -> FilingState:
        """Perform the move. The only node in the system that writes to disk."""
        root = Path(state["root"])
        source = Path(state["source_path"])
        try:
            destination_dir = resolve_within(root, state["destination"])
            destination_dir.mkdir(parents=True, exist_ok=True)

            if is_within(source.resolve(), destination_dir) and source.parent == destination_dir:
                return {"applied": False, "final_path": str(source),
                        "reason": state.get("reason", "") + " File is already here."}

            target = unique_destination(
                destination_dir, state.get("suggested_filename") or source.name
            )
            # shutil.move falls back to copy+unlink when src and dst are on
            # different volumes, where os.rename would raise EXDEV. On a NAS
            # that is the common case, not the edge case.
            shutil.move(str(source), str(target))
            return {"applied": True, "final_path": str(target)}
        except (PathSecurityError, OSError, shutil.Error) as exc:
            logger.warning("filing apply failed", exc_info=True)
            return {"applied": False, "error": f"Move failed: {exc}"}

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _safe_filename(proposed: str, original: str, ext: str) -> str:
        """Strip anything path-like out of an LLM-proposed filename.

        A model that returns ``../../etc/passwd`` must not be able to redirect
        the write. Taking only ``Path(...).name`` reduces any traversal
        attempt to a plain filename.
        """
        if not proposed:
            return original
        cleaned = Path(proposed.replace("\\", "/")).name
        cleaned = "".join(c for c in cleaned if c not in '<>:"|?*').strip(" .")
        if not cleaned:
            return original
        if ext and not cleaned.lower().endswith(ext):
            cleaned = f"{Path(cleaned).stem}{ext}"
        return cleaned[:180]
