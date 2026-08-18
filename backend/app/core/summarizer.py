"""Hierarchical folder summarisation (bottom-up map-reduce).

Why not "feed the whole tree to the LLM"? A 5,000-file share serialises to
far more tokens than any 8k-context local model can hold, and cost grows
linearly with tree size.

Instead we reduce over the tree in post-order:

    leaf folder  --map-->    summary from its own file names + snippets
    parent       --reduce--> summary from its children's summaries + own files

Each LLM call sees a bounded prompt (a handful of child summaries plus a
sample of filenames), so token usage per call is O(1) and total cost is O(D)
calls for D directories -- with ``max_folder_summaries`` as a hard ceiling.
This is the same shape as LangChain's map-reduce chain, specialised to a tree
so that a parent inherits its children's meaning instead of re-reading them.

Summaries are themselves embedded and stored, which is what lets a user ask
"which folder holds the vendor contracts?" and get a *folder* back rather than
a page fragment.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.config import Settings
from app.core.extractors import ExtractorRegistry
from app.core.llm import LLMUnavailableError, complete, embed_documents
from app.core.manifest import Manifest
from app.core.paths import human_size, root_id
from app.core.scanner import DirNode, iter_dirs
from app.core.textutils import collapse_whitespace, truncate
from app.core.vectorstore import (
    KIND_FOLDER_SUMMARY,
    VectorRecord,
    VectorRepository,
    sanitize_metadata,
)

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a meticulous information architect. You describe what a folder "
    "contains so a colleague can decide whether to open it. Be concrete and "
    "specific: name the document types, topics, projects and time periods you "
    "actually see. Never invent content. Never use filler like 'this folder "
    "contains various files'."
)

_TEMPLATE = """Folder: {folder}
Direct files: {direct_files} | Files including subfolders: {total_files} | Size: {size}

Sub-folder summaries:
{child_summaries}

Sample file names:
{file_names}

Content snippets:
{snippets}

Write 1-3 sentences (max 60 words) describing what this folder is for and what
is in it. Start directly with the description - no preamble, no bullet points."""


class FolderSummarizer:
    def __init__(self, settings: Settings, registry: ExtractorRegistry) -> None:
        self._settings = settings
        self._registry = registry

    def summarize_tree(
        self,
        root: Path,
        tree: DirNode,
        repository: VectorRepository,
        manifest: Manifest,
        changed_dirs: set[str] | None = None,
        cancel: threading.Event | None = None,
    ) -> int:
        """Summarise directories bottom-up; returns the number of LLM calls.

        ``changed_dirs`` restricts work to folders whose subtree actually
        changed, reusing the cached summary everywhere else.
        """
        settings = self._settings
        budget = settings.max_folder_summaries
        produced = 0
        fresh: dict[str, str] = {}

        # Post-order guarantees children are summarised before their parent.
        for node in iter_dirs(tree):
            if cancel is not None and cancel.is_set():
                break

            cached = manifest.folder_summaries.get(node.rel_path)
            needs_work = changed_dirs is None or node.rel_path in changed_dirs

            if node.total_files < settings.min_files_for_summary:
                node.summary = cached
                continue
            if not needs_work and cached:
                node.summary = cached
                continue
            if budget <= 0:
                node.summary = cached or self._fallback_summary(node)
                continue

            try:
                summary = self._summarize_node(root, node)
            except LLMUnavailableError as exc:
                logger.warning("LLM unavailable during summarisation: %s", exc)
                node.summary = cached or self._fallback_summary(node)
                break
            except Exception:  # noqa: BLE001
                logger.warning("summary failed for %s", node.rel_path, exc_info=True)
                node.summary = cached or self._fallback_summary(node)
                continue

            node.summary = summary
            fresh[node.rel_path] = summary
            manifest.folder_summaries[node.rel_path] = summary
            budget -= 1
            produced += 1

        # Drop summaries for folders that no longer exist.
        live = {node.rel_path for node in iter_dirs(tree)}
        for rel in list(manifest.folder_summaries):
            if rel not in live:
                manifest.folder_summaries.pop(rel, None)

        if fresh:
            self._store(root, tree, repository, fresh)
        return produced

    # ------------------------------------------------------------- one folder
    def _summarize_node(self, root: Path, node: DirNode) -> str:
        settings = self._settings

        child_summaries = "\n".join(
            f"- {child.name}/: {child.summary}"
            for child in node.children[:8]
            if child.summary
        ) or "(none)"

        sample = sorted(node.files, key=lambda f: f.size_bytes, reverse=True)[
            : settings.summary_sample_files
        ]
        file_names = "\n".join(f"- {f.name} ({human_size(f.size_bytes)})" for f in sample) or "(none)"

        snippets: list[str] = []
        for record in sample[:4]:
            if not self._registry.supports(record.ext):
                continue
            text, _ = self._registry.extract(
                Path(record.path), settings.summary_snippet_chars
            )
            if text:
                snippets.append(
                    f"[{record.name}] {truncate(collapse_whitespace(text), 400, '...')}"
                )
        snippet_block = "\n".join(snippets) or "(no extractable text)"

        prompt = _TEMPLATE.format(
            folder=node.rel_path or "(root)",
            direct_files=len(node.files),
            total_files=node.total_files,
            size=human_size(node.total_size),
            child_summaries=child_summaries,
            file_names=file_names,
            snippets=snippet_block,
        )
        return truncate(collapse_whitespace(complete(_SYSTEM, prompt)), 600, "")

    @staticmethod
    def _fallback_summary(node: DirNode) -> str:
        """Deterministic description used when the LLM is unreachable.

        Graceful degradation: the tree view still shows something useful
        instead of an empty column.
        """
        extensions: dict[str, int] = {}
        for record in node.files:
            extensions[record.ext or "(no ext)"] = extensions.get(record.ext or "(no ext)", 0) + 1
        top = sorted(extensions.items(), key=lambda kv: -kv[1])[:4]
        breakdown = ", ".join(f"{count} x {ext}" for ext, count in top) or "no files"
        return (
            f"{node.total_files} files ({human_size(node.total_size)}) "
            f"across {len(node.children)} sub-folders - {breakdown}."
        )

    # ------------------------------------------------------------------ store
    def _store(
        self,
        root: Path,
        tree: DirNode,
        repository: VectorRepository,
        fresh: dict[str, str],
    ) -> None:
        nodes = {node.rel_path: node for node in iter_dirs(tree)}
        texts: list[str] = []
        records: list[VectorRecord] = []

        for rel, summary in fresh.items():
            node = nodes.get(rel)
            if node is None:
                continue
            label = rel or "(root)"
            texts.append(f"Folder: {label}\n\n{summary}")
            records.append(
                VectorRecord(
                    id=f"{root_id(root)}:folder:{rel}",
                    document=summary,
                    embedding=[],
                    metadata=sanitize_metadata(
                        {
                            "kind": KIND_FOLDER_SUMMARY,
                            "rel_path": rel,
                            "abs_path": node.abs_path,
                            "name": node.name,
                            "ext": "folder",
                            "total_files": node.total_files,
                            "total_size": node.total_size,
                        }
                    ),
                )
            )

        if not records:
            return
        vectors = embed_documents(texts, self._settings.embedding_batch_size)
        for record, embedding in zip(records, vectors):
            record.embedding = embedding
        repository.upsert(root, records)
