"""Deterministic cleanup analysis.

Deliberately *not* LLM-driven. "Are these two files byte-identical?" has an
exact answer, and an exact answer is what a user needs before deleting
anything. The LLM is used only at the end, to narrate findings that were
already computed. The analyser never mutates the filesystem -- it returns
suggestions and the client decides.

Duplicate detection is the interesting part. Hashing every file is O(total
bytes) and dominates on a large share. Instead we exploit a one-way
implication: identical files must have identical sizes.

    1. Group by size.                       O(N), stat data already in hand.
    2. Discard singleton groups.            No same-size peer => not a dupe.
    3. Hash only survivors, cheap-first:
       a. 4 KiB head digest to split the group further,
       b. full SHA-256 only where heads still collide.

On a typical share step 2 removes the large majority of files and step 3a
removes most of the rest, so full hashing touches a small fraction of the
data while remaining exact.
"""

from __future__ import annotations

import heapq
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import Settings
from app.core.llm import LLMUnavailableError, acomplete
from app.core.paths import human_size
from app.core.scanner import FileRecord, file_digest

logger = logging.getLogger(__name__)

_HEAD_BYTES = 4096

_VERSIONED_NAME_RE = re.compile(
    r"""^(?P<base>.+?)
        (?:
            \s*[-_ ]?\(\d+\)            |   # report (1).pdf
            \s*[-_ ]?copy(?:\s*\d+)?    |   # report copy 2.pdf
            \s*[-_ ]?v\d+               |   # report_v3.pdf
            \s*[-_ ](?:final|latest|new|old|draft)\d*
        )$
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(slots=True)
class DuplicateGroup:
    digest: str
    size_bytes: int
    wasted_bytes: int
    paths: list[str]
    names: list[str]


@dataclass(slots=True)
class Finding:
    rel_path: str
    reason: str
    size_bytes: int
    modified_at: float


@dataclass(slots=True)
class CleanupReport:
    root: str
    files_examined: int
    total_size_bytes: int
    reclaimable_bytes: int
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)
    near_duplicates: list[Finding] = field(default_factory=list)
    empty_files: list[Finding] = field(default_factory=list)
    junk_files: list[Finding] = field(default_factory=list)
    stale_files: list[Finding] = field(default_factory=list)
    large_files: list[Finding] = field(default_factory=list)
    empty_dirs: list[str] = field(default_factory=list)
    narrative: str = ""

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "filesExamined": self.files_examined,
            "totalSizeBytes": self.total_size_bytes,
            "reclaimableBytes": self.reclaimable_bytes,
            "reclaimableHuman": human_size(self.reclaimable_bytes),
            "duplicateGroups": [asdict(g) for g in self.duplicate_groups],
            "nearDuplicates": [asdict(f) for f in self.near_duplicates],
            "emptyFiles": [asdict(f) for f in self.empty_files],
            "junkFiles": [asdict(f) for f in self.junk_files],
            "staleFiles": [asdict(f) for f in self.stale_files],
            "largeFiles": [asdict(f) for f in self.large_files],
            "emptyDirs": self.empty_dirs,
            "narrative": self.narrative,
        }


class CleanupAnalyzer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def analyze(self, root: Path, files: list[FileRecord]) -> CleanupReport:
        settings = self._settings
        now = time.time()
        stale_cutoff = now - settings.stale_days * 86_400

        duplicates = self._find_duplicates(files)
        reclaimable = sum(group.wasted_bytes for group in duplicates)

        junk: list[Finding] = []
        empty: list[Finding] = []
        stale: list[Finding] = []

        for record in files:
            if (
                record.name.lower() in settings.junk_names
                or record.ext in settings.junk_exts
                or record.name.startswith("~$")
            ):
                junk.append(self._finding(record, "Temporary or OS-generated file"))
                reclaimable += record.size_bytes
            elif record.size_bytes == 0:
                empty.append(self._finding(record, "Zero-byte file"))
            elif record.modified_at < stale_cutoff:
                age_days = int((now - record.modified_at) / 86_400)
                stale.append(self._finding(record, f"Untouched for {age_days} days"))

        return CleanupReport(
            root=str(root),
            files_examined=len(files),
            total_size_bytes=sum(f.size_bytes for f in files),
            reclaimable_bytes=reclaimable,
            duplicate_groups=duplicates,
            near_duplicates=self._find_near_duplicates(files),
            empty_files=empty[:100],
            junk_files=junk[:100],
            stale_files=sorted(stale, key=lambda f: f.modified_at)[:100],
            large_files=self._find_largest(files, settings.large_file_top_k),
            empty_dirs=self._find_empty_dirs(root),
        )

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _finding(record: FileRecord, reason: str) -> Finding:
        return Finding(
            rel_path=record.rel_path,
            reason=reason,
            size_bytes=record.size_bytes,
            modified_at=record.modified_at,
        )

    def _find_duplicates(self, files: list[FileRecord]) -> list[DuplicateGroup]:
        by_size: dict[int, list[FileRecord]] = defaultdict(list)
        for record in files:
            if record.size_bytes > 0:
                by_size[record.size_bytes].append(record)

        candidates = [group for group in by_size.values() if len(group) > 1]
        if not candidates:
            return []

        groups: list[DuplicateGroup] = []
        for same_size in candidates:
            for head_group in self._split_by_head(same_size).values():
                if len(head_group) < 2:
                    continue
                by_digest: dict[str, list[FileRecord]] = defaultdict(list)
                for record in head_group:
                    digest = file_digest(
                        Path(record.path), self._settings.hash_read_block_bytes
                    )
                    if digest:
                        by_digest[digest].append(record)

                for digest, members in by_digest.items():
                    if len(members) < 2:
                        continue
                    members.sort(key=lambda r: r.modified_at)
                    size = members[0].size_bytes
                    groups.append(
                        DuplicateGroup(
                            digest=digest[:16],
                            size_bytes=size,
                            wasted_bytes=size * (len(members) - 1),
                            paths=[m.rel_path for m in members],
                            names=[m.name for m in members],
                        )
                    )

        groups.sort(key=lambda g: -g.wasted_bytes)
        return groups[:100]

    @staticmethod
    def _split_by_head(records: list[FileRecord]) -> dict[bytes, list[FileRecord]]:
        """Partition same-size files by their first 4 KiB.

        One short read per file rules out most false pairings before we pay
        for a full-file hash.
        """
        buckets: dict[bytes, list[FileRecord]] = defaultdict(list)
        for record in records:
            try:
                with open(record.path, "rb") as handle:
                    head = handle.read(_HEAD_BYTES)
            except OSError:
                continue
            buckets[head].append(record)
        return buckets

    @staticmethod
    def _find_near_duplicates(files: list[FileRecord]) -> list[Finding]:
        """Name-pattern siblings: ``report.pdf`` next to ``report (1).pdf``.

        Content may legitimately differ, so these are flagged for review
        rather than counted as reclaimable space.
        """
        by_base: dict[tuple[str, str, str], list[FileRecord]] = defaultdict(list)
        for record in files:
            match = _VERSIONED_NAME_RE.match(record.stem)
            base = (match.group("base") if match else record.stem).strip().lower()
            by_base[(record.parent_rel, base, record.ext)].append(record)

        findings: list[Finding] = []
        for (_, base, _), members in by_base.items():
            if len(members) < 2:
                continue
            members.sort(key=lambda r: r.modified_at, reverse=True)
            for stale_member in members[1:]:
                findings.append(
                    Finding(
                        rel_path=stale_member.rel_path,
                        reason=f"Looks like an older revision of '{base}'",
                        size_bytes=stale_member.size_bytes,
                        modified_at=stale_member.modified_at,
                    )
                )
        return sorted(findings, key=lambda f: -f.size_bytes)[:60]

    @staticmethod
    def _find_largest(files: list[FileRecord], k: int) -> list[Finding]:
        """Top-K by size via a heap: O(N log K) time, O(K) space.

        Cheaper than sorting all N when K is small, which it always is here.
        """
        largest = heapq.nlargest(k, files, key=lambda r: r.size_bytes)
        return [
            Finding(
                rel_path=r.rel_path,
                reason=f"Large file ({human_size(r.size_bytes)})",
                size_bytes=r.size_bytes,
                modified_at=r.modified_at,
            )
            for r in largest
            if r.size_bytes > 0
        ]

    def _find_empty_dirs(self, root: Path) -> list[str]:
        """Directories with no files anywhere beneath them.

        Requires its own walk because the scanner only yields files, so
        a directory containing nothing leaves no trace in the file list.
        """
        empty: list[str] = []
        ignored = self._settings.ignored_dirs
        # topdown=True lets us prune `dirnames` in place, which os.walk honours
        # as the set of directories still to visit.
        for current, dirnames, filenames in os.walk(root, topdown=True, onerror=None):
            dirnames[:] = [d for d in dirnames if d.lower() not in ignored]
            current_path = Path(current)
            if not filenames and not dirnames and current_path != root:
                empty.append(current_path.relative_to(root).as_posix())
            if len(empty) >= 100:
                break
        return empty

    # ------------------------------------------------------------- narrative
    async def narrate(self, report: CleanupReport) -> str:
        """Turn the computed findings into an actionable paragraph."""
        lines = [
            f"Files examined: {report.files_examined}",
            f"Total size: {human_size(report.total_size_bytes)}",
            f"Reclaimable: {human_size(report.reclaimable_bytes)}",
            f"Exact-duplicate groups: {len(report.duplicate_groups)}",
            f"Probable older revisions: {len(report.near_duplicates)}",
            f"Zero-byte files: {len(report.empty_files)}",
            f"Temp/OS junk: {len(report.junk_files)}",
            f"Stale (> {self._settings.stale_days} days): {len(report.stale_files)}",
            f"Empty directories: {len(report.empty_dirs)}",
        ]
        for group in report.duplicate_groups[:5]:
            lines.append(
                f"- duplicate x{len(group.paths)} ({human_size(group.size_bytes)} each): "
                + ", ".join(group.paths[:3])
            )

        try:
            return await acomplete(
                "You are a pragmatic digital-archivist. Given cleanup statistics, "
                "write a short, prioritised recommendation. Be specific about which "
                "action reclaims the most space. Never suggest deleting something "
                "the data does not support. Max 120 words, plain prose.",
                "\n".join(lines),
            )
        except LLMUnavailableError as exc:
            logger.info("cleanup narrative skipped: %s", exc)
            return (
                f"{len(report.duplicate_groups)} duplicate groups and "
                f"{len(report.junk_files)} junk files account for about "
                f"{human_size(report.reclaimable_bytes)} of reclaimable space. "
                "(Start Ollama for a written recommendation.)"
            )
