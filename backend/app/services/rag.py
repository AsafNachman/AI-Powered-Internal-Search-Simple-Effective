"""Retrieval-augmented question answering over an indexed folder.

Pipeline
--------
1. **Plan.** Split the natural-language question into a semantic part and hard
   metadata constraints ("*pdf* invoices from *last quarter*"). Filters are
   applied in the vector store's ``where`` clause, so they prune the candidate
   set before the ANN search instead of after it.
2. **Retrieve.** Over-fetch ``fetch_k`` neighbours, then re-rank.
3. **Diversify.** Cap chunks per file so one verbose document cannot crowd out
   every other answer -- the cheap, deterministic cousin of MMR.
4. **Generate.** Feed numbered sources to the LLM under a strict grounding
   instruction and stream the answer back with ``[n]`` citations.

Step 1 uses the LLM with a deterministic regex planner as fallback, so search
still works (slightly less precisely) when Ollama is down or returns junk.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.config import Settings
from app.core.llm import (
    LLMUnavailableError,
    acomplete,
    build_messages,
    describe_ollama_error,
    embed_query,
    get_chat_model,
)
from app.core.textutils import extract_json, message_text, truncate
from app.core.vectorstore import (
    KIND_FOLDER_SUMMARY,
    RetrievedChunk,
    VectorRepository,
)

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM = """You are an internal-search assistant for a private file share.

Rules:
- Answer ONLY from the numbered sources provided. They are the entire world.
- Cite every claim with the source number in square brackets, e.g. [2].
- If the sources do not contain the answer, say exactly what is missing and
  suggest a better query. Do not guess and do not use outside knowledge.
- Mention file names and folders explicitly so the user can locate documents.
- Be concise and direct. Use short paragraphs or a tight list."""

_PLANNER_SYSTEM = """You convert a file-search question into a JSON search plan.

Return ONLY a JSON object with these keys:
  "semantic_query": string  - the conceptual part of the question, no filters
  "extensions":     array   - file extensions with the dot, e.g. [".pdf"]; [] if unspecified
  "name_contains":  string  - a literal filename fragment, or "" if unspecified
  "days_back":      integer - recency window in days, 0 if unspecified
  "target":         string  - "files" or "folders"

No prose, no code fence. Just the object."""

_EXT_ALIASES: dict[str, list[str]] = {
    "pdf": [".pdf"],
    "pdfs": [".pdf"],
    "word": [".docx"],
    "doc": [".docx"],
    "docx": [".docx"],
    "excel": [".xlsx", ".xlsm", ".csv"],
    "spreadsheet": [".xlsx", ".xlsm", ".csv"],
    "spreadsheets": [".xlsx", ".xlsm", ".csv"],
    "csv": [".csv"],
    "powerpoint": [".pptx"],
    "slide": [".pptx"],
    "slides": [".pptx"],
    "deck": [".pptx"],
    "markdown": [".md"],
    "image": [".png", ".jpg", ".jpeg", ".gif", ".webp"],
    "images": [".png", ".jpg", ".jpeg", ".gif", ".webp"],
}

_RECENCY_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\btoday\b", re.I), 1),
    (re.compile(r"\byesterday\b", re.I), 2),
    (re.compile(r"\b(this|past|last)\s+week\b", re.I), 7),
    (re.compile(r"\b(this|past|last)\s+month\b", re.I), 31),
    (re.compile(r"\b(this|past|last)\s+quarter\b", re.I), 92),
    (re.compile(r"\b(this|past|last)\s+year\b", re.I), 365),
    (re.compile(r"\blast\s+(\d{1,3})\s+days?\b", re.I), -1),
]

_FOLDER_HINT_RE = re.compile(r"\b(folder|directory|where.*(stored|kept|live))\b", re.I)
_QUOTED_RE = re.compile(r"[\"'\u201c]([^\"'\u201d]{2,60})[\"'\u201d]")
# Unquoted fallback for "file named X.ext" / "file called X.ext" -- users
# often skip quotes entirely, and a filename almost always ends in an
# extension, which anchors the match reliably.
_NAMED_FILE_RE = re.compile(
    r"\b(?:named|called|titled)\s+([\w][\w .,()\-]{0,80}\.[A-Za-z0-9]{1,6})", re.I
)


@dataclass(slots=True)
class SearchPlan:
    semantic_query: str
    extensions: list[str] = field(default_factory=list)
    name_contains: str = ""
    days_back: int = 0
    target: str = "files"

    def to_dict(self) -> dict[str, Any]:
        return {
            "semanticQuery": self.semantic_query,
            "extensions": self.extensions,
            "nameContains": self.name_contains,
            "daysBack": self.days_back,
            "target": self.target,
        }

    def to_where(self) -> dict[str, Any] | None:
        """Compile the plan into a Chroma metadata filter.

        Chroma requires multiple predicates to be wrapped in an explicit
        ``$and``; a bare multi-key dict is rejected.
        """
        clauses: list[dict[str, Any]] = []
        if self.extensions:
            clauses.append({"ext": {"$in": [e.lower() for e in self.extensions]}})
        if self.days_back > 0:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=self.days_back)
            ).timestamp()
            clauses.append({"modified_at": {"$gte": cutoff}})
        if self.target == "folders":
            clauses.append({"kind": KIND_FOLDER_SUMMARY})

        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}


@dataclass(slots=True)
class SearchResult:
    plan: SearchPlan
    chunks: list[RetrievedChunk]
    took_ms: float

    def sources_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "index": position,
                "relPath": chunk.rel_path,
                "absPath": chunk.abs_path,
                "name": chunk.metadata.get("name", Path(chunk.rel_path).name),
                "ext": chunk.metadata.get("ext", ""),
                "kind": chunk.metadata.get("kind", ""),
                "score": round(chunk.score, 4),
                "modifiedAt": chunk.metadata.get("modified_at"),
                "size": chunk.metadata.get("size"),
                "excerpt": truncate(chunk.document, 500, "..."),
            }
            for position, chunk in enumerate(self.chunks, start=1)
        ]


class QueryPlanner:
    """LLM-first plan extraction with a deterministic regex fallback."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def plan(self, question: str) -> SearchPlan:
        heuristic = self.heuristic_plan(question)
        if not self._settings.enable_llm_query_planner:
            return heuristic
        try:
            raw = await acomplete(_PLANNER_SYSTEM, question)
        except LLMUnavailableError:
            return heuristic

        payload = extract_json(raw, dict)
        if not payload:
            return heuristic

        try:
            plan = SearchPlan(
                semantic_query=str(payload.get("semantic_query") or question).strip(),
                extensions=[
                    e if e.startswith(".") else f".{e}"
                    for e in payload.get("extensions") or []
                    if isinstance(e, str) and e.strip()
                ],
                name_contains=str(payload.get("name_contains") or "").strip(),
                days_back=max(0, int(payload.get("days_back") or 0)),
                target="folders" if payload.get("target") == "folders" else "files",
            )
        except (TypeError, ValueError):
            return heuristic

        # Trust the deterministic parser over the model where it found
        # something concrete: regexes do not hallucinate a ".xlsx".
        if heuristic.extensions and not plan.extensions:
            plan.extensions = heuristic.extensions
        if heuristic.days_back and not plan.days_back:
            plan.days_back = heuristic.days_back
        if not plan.semantic_query:
            plan.semantic_query = question
        return plan

    @staticmethod
    def heuristic_plan(question: str) -> SearchPlan:
        lowered = question.lower()

        extensions: list[str] = []
        for token in re.findall(r"[a-z]+", lowered):
            for extension in _EXT_ALIASES.get(token, []):
                if extension not in extensions:
                    extensions.append(extension)
        for literal in re.findall(r"\.([a-z0-9]{1,5})\b", lowered):
            candidate = f".{literal}"
            if candidate not in extensions and literal.isalpha():
                extensions.append(candidate)

        days_back = 0
        for pattern, window in _RECENCY_PATTERNS:
            match = pattern.search(question)
            if not match:
                continue
            days_back = int(match.group(1)) if window == -1 else window
            break

        name_contains = ""
        quoted = _QUOTED_RE.search(question)
        if quoted:
            name_contains = quoted.group(1)
        else:
            named = _NAMED_FILE_RE.search(question)
            if named:
                name_contains = named.group(1).strip().rstrip("?.!,;:")

        return SearchPlan(
            semantic_query=question.strip(),
            extensions=extensions,
            name_contains=name_contains,
            days_back=days_back,
            target="folders" if _FOLDER_HINT_RE.search(question) else "files",
        )


class RagService:
    def __init__(self, settings: Settings, repository: VectorRepository) -> None:
        self._settings = settings
        self._repository = repository
        self._planner = QueryPlanner(settings)

    # ------------------------------------------------------------- retrieval
    async def search(self, root: Path, question: str, top_k: int | None = None) -> SearchResult:
        started = time.perf_counter()
        settings = self._settings
        plan = await self._planner.plan(question)
        limit = top_k or settings.retrieval_top_k

        # embed_query and the Chroma query are both blocking (an HTTP round
        # trip to Ollama, then disk I/O), so they must not run directly on
        # the event loop -- doing so would stall every other request (health
        # checks, the tree view, a background watcher's reindex) for as long
        # as this call takes, which on a busy/CPU box is long enough to look
        # like a hung connection to the browser.
        embedding = await asyncio.to_thread(embed_query, plan.semantic_query or question)
        chunks = await asyncio.to_thread(
            self._repository.query, root, embedding, settings.retrieval_fetch_k, plan.to_where()
        )

        # A filter that matches nothing is worse than no filter: retry
        # unconstrained rather than telling the user "not found".
        if not chunks and plan.to_where() is not None:
            chunks = await asyncio.to_thread(
                self._repository.query, root, embedding, settings.retrieval_fetch_k
            )

        # Hybrid retrieval: a literal filename fragment is exact-match
        # evidence that dense vector search can miss entirely -- especially
        # for tiny/empty files whose embedding carries almost no content
        # signal. Keyword hits are merged into the candidate set *before*
        # ranking so they compete on their own lexical score instead of only
        # boosting a chunk that the ANN search happened to find anyway.
        if plan.name_contains:
            keyword_hits = await asyncio.to_thread(
                self._repository.find_by_name, root, plan.name_contains
            )
            seen_ids = {chunk.id for chunk in chunks}
            chunks.extend(hit for hit in keyword_hits if hit.id not in seen_ids)

        chunks = self._post_rank(chunks, plan)
        return SearchResult(
            plan=plan,
            chunks=chunks[:limit],
            took_ms=(time.perf_counter() - started) * 1000,
        )

    def _post_rank(self, chunks: list[RetrievedChunk], plan: SearchPlan) -> list[RetrievedChunk]:
        settings = self._settings
        needle = plan.name_contains.lower()

        boosted: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.score < settings.min_relevance_score:
                continue
            score = chunk.score
            # Lexical nudge: an exact filename match is strong evidence that
            # pure vector similarity underweights. Capped at 1.0 so a boosted
            # weak match can never outrank a genuine semantic hit outright.
            if needle and needle in str(chunk.metadata.get("name", "")).lower():
                score = min(1.0, score + 0.25)
            if chunk.metadata.get("kind") == KIND_FOLDER_SUMMARY and plan.target == "folders":
                score = min(1.0, score + 0.1)
            chunk.score = score
            boosted.append(chunk)

        boosted.sort(key=lambda c: c.score, reverse=True)

        per_file: dict[str, int] = {}
        diversified: list[RetrievedChunk] = []
        for chunk in boosted:
            key = chunk.rel_path
            if per_file.get(key, 0) >= settings.max_chunks_per_file:
                continue
            per_file[key] = per_file.get(key, 0) + 1
            diversified.append(chunk)
        return diversified

    # ------------------------------------------------------------ generation
    @staticmethod
    def build_prompt(question: str, result: SearchResult) -> str:
        if not result.chunks:
            return (
                f"Question: {question}\n\n"
                "SOURCES: (none matched)\n\n"
                "Tell the user no indexed content matched, and suggest two more "
                "specific phrasings."
            )

        blocks: list[str] = []
        for position, chunk in enumerate(result.chunks, start=1):
            label = "FOLDER" if chunk.metadata.get("kind") == KIND_FOLDER_SUMMARY else "FILE"
            modified = chunk.metadata.get("modified_at")
            when = (
                datetime.fromtimestamp(float(modified), tz=timezone.utc).strftime("%Y-%m-%d")
                if isinstance(modified, int | float)
                else "unknown"
            )
            rel_path = chunk.rel_path or "(root)"
            # Filename and path are on their own labelled line, not just
            # implied by rel_path, so the model cannot fail to notice a file
            # exists even when its content is empty or irrelevant to the
            # question.
            name = chunk.metadata.get("name") or Path(rel_path).name
            blocks.append(
                f"[{position}] {label}\n"
                f"Filename: {name}\n"
                f"Filepath: {rel_path}\n"
                f"Modified: {when} | Relevance: {chunk.score:.2f}\n\n"
                f"{chunk.document}"
            )

        return (
            f"Question: {question}\n\n"
            f"SOURCES:\n{'-' * 60}\n"
            + f"\n{'-' * 60}\n".join(blocks)
            + f"\n{'-' * 60}\n\nAnswer the question using only these sources, with [n] citations."
        )

    async def answer_stream(self, question: str, result: SearchResult) -> AsyncIterator[str]:
        """Yield answer tokens as the local model produces them.

        Streaming matters disproportionately here: an 8B model on CPU may need
        20+ seconds for a full answer, but first-token latency is ~1s. The
        user sees progress instead of a spinner.
        """
        messages = build_messages(_ANSWER_SYSTEM, self.build_prompt(question, result))
        started = time.perf_counter()
        first_token_at: float | None = None
        try:
            async for chunk in get_chat_model().astream(messages):
                text = message_text(chunk)
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                        logger.info(
                            "first token in %.2fs (model=%s)",
                            first_token_at - started,
                            self._settings.chat_model,
                        )
                    yield text
        except Exception as exc:  # noqa: BLE001 - reraise with an actionable message
            detail = describe_ollama_error(exc, self._settings)
            logger.error("answer_stream failed after %.2fs: %s", time.perf_counter() - started, detail)
            raise RuntimeError(detail) from exc
