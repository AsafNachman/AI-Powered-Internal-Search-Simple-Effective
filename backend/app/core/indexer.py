"""The ingestion pipeline (Facade over scan -> extract -> chunk -> embed -> store).

    scan  ->  diff vs manifest  ->  extract  ->  chunk  ->  embed  ->  upsert
                                                                        |
                                                        hierarchical folder summaries

The diff step is what makes repeat runs cheap: only added or modified files
reach the (expensive) extract/embed stages, and vectors for deleted files are
purged so search never cites a file that no longer exists.

Cost model for one run over N files, M of them changed:
* scan            O(N) syscalls
* diff            O(N) hash lookups
* extract/chunk   O(bytes in M)
* embed           ceil(C / batch) HTTP calls for C chunks -- the wall-clock
                  bottleneck, and the reason the diff exists
* upsert          O(C)
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.core.extractors import ExtractorRegistry
from app.core.llm import embed_documents
from app.core.manifest import FileState, Manifest, ManifestStore
from app.core.paths import root_id
from app.core.scanner import DirectoryScanner, FileRecord, ScanResult
from app.core.summarizer import FolderSummarizer
from app.core.textutils import truncate
from app.core.vectorstore import (
    KIND_CHUNK,
    VectorRecord,
    VectorRepository,
    sanitize_metadata,
)

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


@dataclass(slots=True)
class IndexReport:
    root: str
    root_id: str
    total_files: int
    indexed_files: int
    skipped_unchanged: int
    removed_files: int
    chunks_written: int
    folders_summarized: int
    directories: int
    total_size_bytes: int
    duration_s: float
    truncated: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "rootId": self.root_id,
            "totalFiles": self.total_files,
            "indexedFiles": self.indexed_files,
            "skippedUnchanged": self.skipped_unchanged,
            "removedFiles": self.removed_files,
            "chunksWritten": self.chunks_written,
            "foldersSummarized": self.folders_summarized,
            "directories": self.directories,
            "totalSizeBytes": self.total_size_bytes,
            "durationSeconds": round(self.duration_s, 2),
            "truncated": self.truncated,
            "warnings": self.warnings,
        }


class IndexingService:
    def __init__(
        self,
        settings: Settings,
        repository: VectorRepository,
        manifests: ManifestStore,
        registry: ExtractorRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._manifests = manifests
        self._registry = registry or ExtractorRegistry()
        self._scanner = DirectoryScanner(settings)
        self._summarizer = FolderSummarizer(settings, self._registry)
        # RecursiveCharacterTextSplitter walks a separator hierarchy from
        # coarse to fine (paragraph -> line -> sentence -> word), splitting on
        # the largest separator that gets a piece under chunk_size. Semantic
        # units therefore stay intact far more often than with a fixed window.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
            length_function=len,
        )

    # ------------------------------------------------------------------ main
    def index_root(
        self,
        root: Path,
        force: bool = False,
        summarize: bool | None = None,
        progress: ProgressFn | None = None,
        cancel: threading.Event | None = None,
    ) -> IndexReport:
        started = time.perf_counter()
        report_progress = progress or (lambda _m, _p: None)
        warnings: list[str] = []

        report_progress("Scanning directory tree...", 0.02)
        scan: ScanResult = self._scanner.scan(root, cancel=cancel)
        if cancel is not None and cancel.is_set():
            raise IndexCancelled()
        if scan.truncated:
            warnings.append(
                f"Scan stopped at the {self._settings.max_files_per_index} file cap."
            )

        manifest = Manifest(root=str(root)) if force else self._manifests.load(root)
        if force:
            self._repository.drop(root)

        current = {record.rel_path: record for record in scan.files}
        changed, unchanged = self._diff(current, manifest)
        removed = [rel for rel in manifest.files if rel not in current]

        report_progress(
            f"{len(scan.files)} files found - {len(changed)} to index, "
            f"{len(unchanged)} unchanged.",
            0.08,
        )

        if removed:
            self._repository.delete_by_rel_paths(root, removed)
            for rel in removed:
                manifest.files.pop(rel, None)

        chunks_written = 0
        if changed:
            # Replace-then-write: a shrinking file would otherwise leave
            # orphan chunks from its previous, longer revision.
            self._repository.delete_by_rel_paths(root, [r.rel_path for r in changed])
            chunks_written = self._embed_and_store(
                root, changed, manifest, report_progress, cancel
            )

        folders_summarized = 0
        want_summaries = (
            self._settings.summarize_folders if summarize is None else summarize
        )
        if want_summaries and scan.files:
            report_progress("Summarising folders...", 0.85)
            try:
                folders_summarized = self._summarizer.summarize_tree(
                    root=root,
                    tree=scan.tree,
                    repository=self._repository,
                    manifest=manifest,
                    changed_dirs=None if force else self._dirs_touched(changed, removed),
                    cancel=cancel,
                )
            except Exception as exc:  # noqa: BLE001 - summaries are a nice-to-have
                logger.warning("folder summarisation failed", exc_info=True)
                warnings.append(f"Folder summaries unavailable: {exc}")

        manifest.updated_at = time.time()
        manifest.stats = {
            "total_files": len(scan.files),
            "total_size": scan.tree.total_size,
            "directories": scan.directories_scanned,
            "chunks": self._repository.count(root),
        }
        self._manifests.save(root, manifest)
        report_progress("Index ready.", 1.0)

        return IndexReport(
            root=str(root),
            root_id=root_id(root),
            total_files=len(scan.files),
            indexed_files=len(changed),
            skipped_unchanged=len(unchanged),
            removed_files=len(removed),
            chunks_written=chunks_written,
            folders_summarized=folders_summarized,
            directories=scan.directories_scanned,
            total_size_bytes=scan.tree.total_size,
            duration_s=time.perf_counter() - started,
            truncated=scan.truncated,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ diff
    @staticmethod
    def _diff(
        current: dict[str, FileRecord], manifest: Manifest
    ) -> tuple[list[FileRecord], list[str]]:
        changed: list[FileRecord] = []
        unchanged: list[str] = []
        for rel, record in current.items():
            previous = manifest.files.get(rel)
            if previous is not None and previous.matches(
                record.size_bytes, record.modified_at
            ):
                unchanged.append(rel)
            else:
                changed.append(record)
        return changed, unchanged

    @staticmethod
    def _dirs_touched(changed: list[FileRecord], removed: list[str]) -> set[str]:
        """Ancestor closure of every touched file.

        A folder summary is stale if anything *anywhere beneath it* changed,
        so we walk each path upward and mark every ancestor.
        """
        parents = [record.parent_rel for record in changed]
        for rel in removed:
            parent = Path(rel).parent.as_posix()
            # PurePath renders "no parent" as ".", which is not a rel_path we
            # ever store -- the root is the empty string.
            parents.append("" if parent == "." else parent)

        dirs: set[str] = {""}
        for rel in parents:
            parts = [part for part in rel.split("/") if part]
            for depth in range(1, len(parts) + 1):
                dirs.add("/".join(parts[:depth]))
        return dirs

    # ------------------------------------------------------------- embedding
    def _embed_and_store(
        self,
        root: Path,
        records: list[FileRecord],
        manifest: Manifest,
        progress: ProgressFn,
        cancel: threading.Event | None,
    ) -> int:
        settings = self._settings
        total = len(records)
        written = 0
        pending: list[tuple[VectorRecord, str]] = []
        flush_at = max(settings.embedding_batch_size * 4, 64)

        def flush() -> None:
            nonlocal written, pending
            if not pending:
                return
            vectors = embed_documents(
                [text for _, text in pending], settings.embedding_batch_size
            )
            for (record, _), embedding in zip(pending, vectors):
                record.embedding = embedding
            written += self._repository.upsert(root, [r for r, _ in pending])
            pending = []

        for position, file_record in enumerate(records, start=1):
            if cancel is not None and cancel.is_set():
                raise IndexCancelled()

            text, kind = self._registry.extract(
                Path(file_record.path), settings.max_chars_per_file
            )
            chunk_count = 0
            if text:
                for order, chunk in enumerate(self._splitter.split_text(text)):
                    chunk = chunk.strip()
                    if len(chunk) < 24:  # boilerplate fragment, not worth a vector
                        continue
                    pending.append(
                        self._build_record(root, file_record, kind, order, chunk)
                    )
                    chunk_count += 1

            manifest.files[file_record.rel_path] = FileState(
                size=file_record.size_bytes,
                mtime=file_record.modified_at,
                chunks=chunk_count,
                kind=kind,
                indexed_at=time.time(),
            )

            if len(pending) >= flush_at:
                flush()

            if position % 5 == 0 or position == total:
                progress(
                    f"Embedding {position}/{total}: {file_record.rel_path}",
                    0.08 + 0.75 * (position / total),
                )

        flush()
        return written

    def _build_record(
        self,
        root: Path,
        file_record: FileRecord,
        kind: str,
        order: int,
        chunk: str,
    ) -> tuple[VectorRecord, str]:
        """Create a chunk vector plus the text that should be embedded.

        The embedded text is *contextualised* -- the file's path and name are
        prepended before embedding. Without it, a chunk reading "Q3 was up
        14%" carries no signal for the query "quarterly revenue report", since
        the words that identify the document live in its filename, not its
        body. The stored ``document`` keeps the raw chunk so citations show
        the real text.
        """
        chunk_id = f"{root_id(root)}:{file_record.rel_path}:{order}"
        folder = file_record.parent_rel or "/"
        embed_text = (
            f"File: {file_record.name}\n"
            f"Folder: {folder}\n"
            f"Type: {kind}\n\n"
            f"{chunk}"
        )
        metadata = sanitize_metadata(
            {
                "kind": KIND_CHUNK,
                "rel_path": file_record.rel_path,
                "abs_path": file_record.path,
                "name": file_record.name,
                "stem": file_record.stem,
                "ext": file_record.ext or "none",
                "parent_rel": file_record.parent_rel,
                "doc_kind": kind,
                "size": file_record.size_bytes,
                "modified_at": file_record.modified_at,
                "chunk_index": order,
            }
        )
        return (
            VectorRecord(
                id=chunk_id,
                document=truncate(chunk, self._settings.chunk_size * 2),
                embedding=[],
                metadata=metadata,
            ),
            embed_text,
        )


class IndexCancelled(RuntimeError):
    """Raised when a caller cancels an in-flight index job."""
