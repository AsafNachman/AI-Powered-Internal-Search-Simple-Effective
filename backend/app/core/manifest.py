"""Per-root index manifest, used to make re-indexing incremental.

Rationale
---------
Re-embedding an unchanged 5,000-file share takes minutes and produces byte-
identical vectors. The manifest records ``(size, mtime, chunks)`` per file so a
second run can diff the filesystem against the last run in O(N) dictionary
lookups and touch only what actually changed.

``mtime`` + ``size`` is a heuristic, not a proof of equality -- an edit that
preserves both would be missed. It is the same trade-off ``make`` and ``rsync``
make by default, and the API exposes ``force=true`` for a full rebuild.

Writes go through a temp file plus :func:`os.replace`, which is atomic on both
POSIX and Windows. A crash mid-write therefore leaves the previous manifest
intact rather than a truncated one.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.core.paths import root_id

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


@dataclass(slots=True)
class FileState:
    size: int
    mtime: float
    chunks: int = 0
    kind: str = ""
    indexed_at: float = 0.0

    def matches(self, size: int, mtime: float, tolerance: float = 1.0) -> bool:
        # FAT32 stores mtime at 2-second granularity and network shares can
        # round; a 1s tolerance avoids spurious re-indexing.
        return self.size == size and abs(self.mtime - mtime) <= tolerance


@dataclass(slots=True)
class Manifest:
    root: str
    version: int = MANIFEST_VERSION
    updated_at: float = 0.0
    files: dict[str, FileState] = field(default_factory=dict)
    folder_summaries: dict[str, str] = field(default_factory=dict)
    stats: dict[str, float | int | str] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "root": self.root,
            "version": self.version,
            "updated_at": self.updated_at,
            "files": {
                rel: {
                    "size": s.size,
                    "mtime": s.mtime,
                    "chunks": s.chunks,
                    "kind": s.kind,
                    "indexed_at": s.indexed_at,
                }
                for rel, s in self.files.items()
            },
            "folder_summaries": self.folder_summaries,
            "stats": self.stats,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "Manifest":
        files = {
            str(rel): FileState(
                size=int(state.get("size", 0)),
                mtime=float(state.get("mtime", 0.0)),
                chunks=int(state.get("chunks", 0)),
                kind=str(state.get("kind", "")),
                indexed_at=float(state.get("indexed_at", 0.0)),
            )
            for rel, state in (payload.get("files") or {}).items()
        }
        return cls(
            root=str(payload.get("root", "")),
            version=int(payload.get("version", MANIFEST_VERSION)),
            updated_at=float(payload.get("updated_at", 0.0)),
            files=files,
            folder_summaries=dict(payload.get("folder_summaries") or {}),
            stats=dict(payload.get("stats") or {}),
        )


class ManifestStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()

    def _path_for(self, root: Path) -> Path:
        return self._settings.manifest_dir / f"{root_id(root)}.json"

    def load(self, root: Path) -> Manifest:
        path = self._path_for(root)
        if not path.exists():
            return Manifest(root=str(root))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("manifest unreadable, starting fresh: %s", path, exc_info=True)
            return Manifest(root=str(root))
        manifest = Manifest.from_json(payload)
        if manifest.version != MANIFEST_VERSION:
            logger.info("manifest schema changed, discarding cache for %s", root)
            return Manifest(root=str(root))
        return manifest

    def save(self, root: Path, manifest: Manifest) -> None:
        path = self._path_for(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".json.tmp")
        with self._lock:
            temp.write_text(
                json.dumps(manifest.to_json(), ensure_ascii=False), encoding="utf-8"
            )
            os.replace(temp, path)  # atomic on POSIX and Windows

    def delete(self, root: Path) -> None:
        self._path_for(root).unlink(missing_ok=True)

    def list_roots(self) -> list[Manifest]:
        manifests: list[Manifest] = []
        for path in sorted(self._settings.manifest_dir.glob("*.json")):
            try:
                manifests.append(
                    Manifest.from_json(json.loads(path.read_text(encoding="utf-8")))
                )
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(manifests, key=lambda m: m.updated_at, reverse=True)
