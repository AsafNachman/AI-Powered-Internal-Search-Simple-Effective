"""Coordinates live filesystem watchers with the existing job pipeline.

Each watched root gets one :class:`RootWatcher`. Once its debounce timer
fires, that callback (running on a watchdog worker thread) hops onto the
asyncio event loop and runs the exact same incremental ``index_root(...)``
call a manual "Index" click makes, through :class:`JobManager` -- so a
watch-triggered run gets the same progress reporting, cancellation and
job-history behaviour as any other index job, and the UI cannot tell the
difference except by the job's ``kind``.

This registry is in-memory and per-process, same as ``JobManager`` -- a
restart forgets which roots were being watched, and the frontend re-enables
watching after its next successful index.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings
from app.core.indexer import IndexingService
from app.core.paths import root_id
from app.core.watcher import RootWatcher
from app.services.jobs import JobManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WatchEntry:
    root: Path
    watcher: RootWatcher
    started_at: float = field(default_factory=time.time)
    last_event_at: float | None = None
    last_indexed_at: float | None = None
    last_job_id: str | None = None
    last_error: str = ""
    consecutive_errors: int = 0
    reindexing: bool = False
    pending: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "rootId": root_id(self.root),
            "watching": True,
            "reindexing": self.reindexing,
            "pendingChanges": self.pending,
            "lastEventAt": self.last_event_at,
            "lastIndexedAt": self.last_indexed_at,
            "lastJobId": self.last_job_id,
            "lastError": self.last_error,
        }


class WatchManager:
    """Starts/stops per-root watchers and runs their debounced re-index jobs."""

    def __init__(self, settings: Settings, indexer: IndexingService, jobs: JobManager) -> None:
        self._settings = settings
        self._indexer = indexer
        self._jobs = jobs
        self._entries: dict[str, WatchEntry] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the server's event loop so watcher-thread callbacks can reach it.

        Must be called once, from inside the loop that will serve requests
        (the FastAPI lifespan is the natural place), before any watcher is
        started.
        """
        self._loop = loop

    # -------------------------------------------------------------- control
    def start(self, root: Path) -> dict[str, Any]:
        key = root_id(root)
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                return existing.to_dict()

            watcher = RootWatcher(self._settings, root, lambda: self._handle_change(key))
            entry = WatchEntry(root=root, watcher=watcher)
            self._entries[key] = entry
            watcher.start()

        logger.info("watching %s for changes", root)
        return entry.to_dict()

    def stop(self, root: Path) -> bool:
        with self._lock:
            entry = self._entries.pop(root_id(root), None)
        if entry is None:
            return False
        entry.watcher.stop()
        logger.info("stopped watching %s", root)
        return True

    def stop_all(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            entry.watcher.stop()

    def status(self, root: Path) -> dict[str, Any] | None:
        entry = self._entries.get(root_id(root))
        return entry.to_dict() if entry else None

    def is_watching(self, root: Path) -> bool:
        return root_id(root) in self._entries

    # ---------------------------------------------------------- fs callback
    def _handle_change(self, key: str) -> None:
        """Runs on the watcher's debounce-timer thread, not the event loop."""
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.last_event_at = time.time()

        loop = self._loop
        if loop is None:
            logger.warning("watch fired before an event loop was bound; dropping")
            return
        if entry.reindexing:
            # A run is already in flight; it will notice ``pending`` when it
            # finishes and go again immediately, so nothing is lost.
            entry.pending = True
            return
        asyncio.run_coroutine_threadsafe(self._reindex(key), loop)

    async def _reindex(self, key: str) -> None:
        entry = self._entries.get(key)
        if entry is None:
            return

        entry.reindexing = True
        try:
            while True:
                entry.pending = False
                job = self._jobs.create(kind="watch", target=str(entry.root))
                entry.last_job_id = job.id
                await self._jobs.run(
                    job, self._indexer.index_root, entry.root, force=False, summarize=None
                )

                if job.status == "failed":
                    entry.consecutive_errors += 1
                    entry.last_error = job.error
                    logger.warning(
                        "watch re-index failed for %s (%d/%d): %s",
                        entry.root,
                        entry.consecutive_errors,
                        self._settings.watch_max_consecutive_errors,
                        job.error,
                    )
                else:
                    entry.consecutive_errors = 0
                    entry.last_error = ""
                    entry.last_indexed_at = time.time()

                if entry.consecutive_errors >= self._settings.watch_max_consecutive_errors:
                    logger.warning(
                        "disabling watch on %s after repeated failures", entry.root
                    )
                    # stop() joins the observer thread (blocking, up to 5s);
                    # this coroutine runs on the event loop, so it must not
                    # block it directly.
                    await asyncio.to_thread(self.stop, entry.root)
                    return

                # More changes arrived while this run was in flight (or while
                # it was still queued behind another job) -- go again rather
                # than waiting for an unrelated future event to notice.
                if not entry.pending or key not in self._entries:
                    return
        finally:
            entry.reindexing = False
