"""Recursive filesystem traversal.

Design notes
------------
* **Iterative, not recursive.** An explicit stack removes any risk of blowing
  Python's ~1000-frame recursion limit on deep trees and makes pruning and
  cancellation trivial.
* **``os.scandir`` over ``os.listdir``.** ``scandir`` yields ``DirEntry``
  objects that carry the ``stat`` data the OS already returned as part of the
  directory read (``FindFirstFile`` on Windows, ``getdents`` on Linux). Using
  ``entry.is_dir()``/``entry.stat()`` therefore costs no extra syscall,
  whereas ``os.stat(path)`` per entry would add one round trip per file --
  the dominant cost on a network share.
* **Symlinks are not followed by default**, which is what makes the traversal
  a tree rather than a possibly-cyclic graph. When following is enabled we
  track visited ``(st_dev, st_ino)`` pairs to break cycles.

Complexity: O(N) syscalls for N entries; peak memory O(B x D) for the frontier
(B = branching factor, D = depth), not O(N).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FileRecord:
    """Flat, serialisable description of one file on disk."""

    path: str
    rel_path: str
    name: str
    ext: str
    parent_rel: str
    size_bytes: int
    modified_at: float
    created_at: float
    depth: int

    @property
    def stem(self) -> str:
        return Path(self.name).stem


@dataclass(slots=True)
class DirNode:
    """Node of the visualisable directory tree with rolled-up aggregates."""

    name: str
    rel_path: str
    abs_path: str
    depth: int
    children: list["DirNode"] = field(default_factory=list)
    files: list[FileRecord] = field(default_factory=list)
    total_files: int = 0
    total_size: int = 0
    summary: str | None = None

    def to_dict(self, include_files: bool = True, max_files: int = 50) -> dict:
        return {
            "name": self.name,
            "relPath": self.rel_path,
            "absPath": self.abs_path,
            "depth": self.depth,
            "directFiles": len(self.files),
            "totalFiles": self.total_files,
            "totalSize": self.total_size,
            "summary": self.summary,
            "children": [c.to_dict(include_files, max_files) for c in self.children],
            "files": (
                [
                    {
                        "name": f.name,
                        "relPath": f.rel_path,
                        "ext": f.ext,
                        "size": f.size_bytes,
                        "modifiedAt": f.modified_at,
                    }
                    for f in sorted(self.files, key=lambda f: f.name.lower())[:max_files]
                ]
                if include_files
                else []
            ),
        }


@dataclass(slots=True)
class ScanResult:
    root: str
    files: list[FileRecord]
    tree: DirNode
    directories_scanned: int
    truncated: bool


class DirectoryScanner:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ------------------------------------------------------------------ walk
    def iter_files(
        self,
        root: Path,
        cancel: threading.Event | None = None,
    ) -> Iterator[FileRecord]:
        """Yield every eligible file beneath ``root`` in breadth-ish order."""
        settings = self._settings
        ignored = settings.ignored_dirs
        root_str = str(root)
        seen_inodes: set[tuple[int, int]] = set()

        stack: list[tuple[str, int]] = [(root_str, 0)]
        emitted = 0

        while stack:
            if cancel is not None and cancel.is_set():
                return
            current, depth = stack.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        if cancel is not None and cancel.is_set():
                            return
                        try:
                            is_dir = entry.is_dir(
                                follow_symlinks=settings.follow_symlinks
                            )
                        except OSError:
                            continue

                        if is_dir:
                            if entry.name.lower() in ignored or depth + 1 > settings.max_scan_depth:
                                continue
                            if settings.follow_symlinks:
                                try:
                                    stat = entry.stat()
                                    key = (stat.st_dev, stat.st_ino)
                                except OSError:
                                    continue
                                if key in seen_inodes:
                                    continue  # cycle guard
                                seen_inodes.add(key)
                            stack.append((entry.path, depth + 1))
                            continue

                        if not settings.follow_symlinks and entry.is_symlink():
                            continue

                        record = self._to_record(entry, root, depth + 1)
                        if record is None:
                            continue
                        yield record
                        emitted += 1
                        if emitted >= settings.max_files_per_index:
                            logger.warning(
                                "scan truncated at %s files under %s",
                                emitted,
                                root,
                            )
                            return
            except PermissionError:
                logger.debug("permission denied: %s", current)
            except OSError:
                logger.debug("unreadable directory: %s", current, exc_info=True)

    def _to_record(self, entry: os.DirEntry[str], root: Path, depth: int) -> FileRecord | None:
        try:
            stat = entry.stat(follow_symlinks=self._settings.follow_symlinks)
        except OSError:
            return None
        if stat.st_size > self._settings.max_file_size_bytes:
            return None

        absolute = Path(entry.path)
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            return None

        parent = relative.parent
        return FileRecord(
            path=entry.path,
            rel_path=relative.as_posix(),
            name=entry.name,
            ext=absolute.suffix.lower(),
            parent_rel=parent.as_posix() if str(parent) != "." else "",
            size_bytes=stat.st_size,
            modified_at=stat.st_mtime,
            # st_birthtime exists on Windows/macOS; on Linux st_ctime is the
            # inode-change time, which is the closest available proxy.
            created_at=getattr(stat, "st_birthtime", stat.st_ctime),
            depth=depth,
        )

    # ------------------------------------------------------------------ tree
    def scan(self, root: Path, cancel: threading.Event | None = None) -> ScanResult:
        """Walk ``root`` once and materialise both the file list and the tree."""
        files = list(self.iter_files(root, cancel=cancel))
        tree = self.build_tree(root, files)
        return ScanResult(
            root=str(root),
            files=files,
            tree=tree,
            directories_scanned=self._count_dirs(tree),
            truncated=len(files) >= self._settings.max_files_per_index,
        )

    @staticmethod
    def build_tree(root: Path, files: list[FileRecord]) -> DirNode:
        """Assemble a ``DirNode`` tree from a flat file list.

        Two passes. The first materialises every directory on each file's path
        (memoised in a dict, so each directory is created once) and attaches
        the file -- O(total path segments), effectively O(N). The second is a
        post-order roll-up of counts and sizes -- O(V) over tree nodes.
        """
        index: dict[str, DirNode] = {
            "": DirNode(name=root.name or str(root), rel_path="", abs_path=str(root), depth=0)
        }

        def ensure(rel: str) -> DirNode:
            node = index.get(rel)
            if node is not None:
                return node
            parent_rel, _, name = rel.rpartition("/")
            parent = ensure(parent_rel)
            node = DirNode(
                name=name,
                rel_path=rel,
                abs_path=str(root / Path(rel)),
                depth=parent.depth + 1,
            )
            parent.children.append(node)
            index[rel] = node
            return node

        for record in files:
            ensure(record.parent_rel).files.append(record)

        def rollup(node: DirNode) -> tuple[int, int]:
            count = len(node.files)
            size = sum(f.size_bytes for f in node.files)
            for child in node.children:
                child_count, child_size = rollup(child)
                count += child_count
                size += child_size
            node.total_files = count
            node.total_size = size
            node.children.sort(key=lambda c: (-c.total_files, c.name.lower()))
            return count, size

        rollup(index[""])
        return index[""]

    @staticmethod
    def _count_dirs(node: DirNode) -> int:
        return 1 + sum(DirectoryScanner._count_dirs(child) for child in node.children)


def iter_dirs(node: DirNode) -> Iterator[DirNode]:
    """Post-order traversal: children are always yielded before their parent.

    Hierarchical summarisation depends on this ordering -- a folder summary is
    written from its children's summaries, so the children must exist first.
    """
    for child in node.children:
        yield from iter_dirs(child)
    yield node


def file_digest(path: Path, block_size: int = 1024 * 1024) -> str | None:
    """Streaming SHA-256 of a file's bytes.

    Streamed in blocks so memory stays O(block_size) regardless of file size.
    SHA-256 rather than MD5: the cost difference is irrelevant next to disk
    I/O, and we avoid a collision-prone digest entirely.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while chunk := handle.read(block_size):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()
