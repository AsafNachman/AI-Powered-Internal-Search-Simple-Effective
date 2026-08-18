"""Live filesystem watching for automatic, incremental re-indexing.

One :class:`RootWatcher` wraps one ``watchdog`` observer over one indexed
root, using the OS's native change-notification API (``ReadDirectoryChangesW``
on Windows, ``FSEvents`` on macOS, ``inotify`` on Linux) rather than polling.

Filesystem events arrive in bursts -- an editor's "save" is often a temp file
write, a delete, and a rename in quick succession -- so the handler does not
react to every event. It resets a short timer on each one and only fires
``on_settled`` once the directory has been quiet for ``watch_debounce_seconds``,
collapsing an arbitrarily large burst into a single re-index.

Everything in this module runs on watchdog's own worker threads. Getting the
resulting re-index back onto the asyncio event loop is the caller's job (see
``app/services/watch.py``), which is why ``on_settled`` is a plain, synchronous
callback rather than a coroutine.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from app.config import Settings

logger = logging.getLogger(__name__)


class _DebouncedHandler(FileSystemEventHandler):
    """Collapses a burst of filesystem events into one delayed callback."""

    def __init__(self, settings: Settings, root: Path, on_settled: Callable[[], None]) -> None:
        self._settings = settings
        self._root = root
        self._on_settled = on_settled
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def _ignored(self, raw_path: str) -> bool:
        """Mirror the scanner's and cleanup's ignore rules.

        Without this, editing a file inside ``.git`` or writing a ``.tmp``
        swap file would trigger a re-index just as readily as a real
        document -- the watcher would never go quiet on an active repo.
        """
        try:
            rel = Path(raw_path).relative_to(self._root)
        except ValueError:
            return True  # outside the root entirely (should not happen)

        settings = self._settings
        parts = rel.parts
        if not parts:
            return True
        if any(part.lower() in settings.ignored_dirs for part in parts[:-1]):
            return True
        name = parts[-1]
        if name.lower() in settings.junk_names:
            return True
        if Path(name).suffix.lower() in settings.junk_exts:
            return True
        return False

    def _touch(self, raw_path: str) -> None:
        if self._ignored(raw_path):
            return
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._settings.watch_debounce_seconds, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        with self._lock:
            self._timer = None
        try:
            self._on_settled()
        except Exception:  # noqa: BLE001 - a callback bug must not kill the watcher thread
            logger.exception("watch callback failed for %s", self._root)

    # watchdog dispatches every one of these on its own worker thread.
    def on_created(self, event: FileSystemEvent) -> None:
        self._touch(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._touch(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._touch(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        self._touch(event.src_path)
        self._touch(getattr(event, "dest_path", "") or event.src_path)

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


class RootWatcher:
    """One ``watchdog`` observer over one root, debounced to a single callback."""

    def __init__(self, settings: Settings, root: Path, on_change: Callable[[], None]) -> None:
        self._handler = _DebouncedHandler(settings, root, on_change)
        self._observer = Observer()
        self._observer.schedule(self._handler, str(root), recursive=True)

    def start(self) -> None:
        self._observer.start()

    def stop(self) -> None:
        self._handler.cancel()
        self._observer.stop()
        # Bounded join: a slow platform backend must never hang the request
        # handler that asked this watcher to stop.
        self._observer.join(timeout=5.0)
