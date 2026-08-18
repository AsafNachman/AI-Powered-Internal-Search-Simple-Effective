"""In-process async job registry for long-running index runs.

Indexing is minutes of blocking disk I/O and HTTP calls to Ollama. Running it
inside a request handler would occupy the event loop and time the client out,
so the endpoint registers a job, hands back an id immediately, and the client
polls.

The work itself runs via ``asyncio.to_thread``. That is the correct primitive
here because the workload is *blocking I/O* (file reads, HTTP): the GIL is
released during those waits, so a worker thread genuinely runs in parallel
with the event loop. CPU-bound work would need a process pool instead.

Cancellation uses a ``threading.Event`` polled by the scanner and indexer,
rather than an attempt to kill the thread. Python has no safe thread-kill;
cooperative cancellation at known-safe points is the only correct approach.

This registry is per-process and in-memory, which suits a single-user desktop
demo. Multiple API workers would need Redis or a database behind the same
interface.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

MAX_RETAINED_JOBS = 50


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Job:
    id: str
    kind: str
    target: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = "Queued"
    result: dict[str, Any] | None = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "status": str(self.status),
            "progress": round(self.progress, 4),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "createdAt": self.created_at,
            "finishedAt": self.finished_at,
            "elapsedSeconds": round((self.finished_at or time.time()) - self.created_at, 1),
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        # Guards the dict against concurrent mutation from the event loop and
        # from progress callbacks firing on worker threads.
        self._lock = threading.Lock()

    def create(self, kind: str, target: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, target=target)
        with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ):
            return False
        job.cancel_event.set()
        job.message = "Cancelling..."
        return True

    def _evict_locked(self) -> None:
        """Bound memory by dropping the oldest finished jobs."""
        if len(self._jobs) <= MAX_RETAINED_JOBS:
            return
        finished = sorted(
            (
                j
                for j in self._jobs.values()
                if j.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)
            ),
            key=lambda j: j.finished_at or j.created_at,
        )
        for job in finished[: len(self._jobs) - MAX_RETAINED_JOBS]:
            self._jobs.pop(job.id, None)

    async def run(
        self,
        job: Job,
        work: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Execute ``work`` on a worker thread, mirroring state onto ``job``.

        ``work`` receives ``progress`` and ``cancel`` keyword arguments.
        """

        def report(message: str, fraction: float) -> None:
            job.message = message
            job.progress = max(0.0, min(1.0, fraction))

        job.status = JobStatus.RUNNING
        job.message = "Starting..."
        try:
            result = await asyncio.to_thread(
                work, *args, progress=report, cancel=job.cancel_event, **kwargs
            )
            if job.cancel_event.is_set():
                job.status = JobStatus.CANCELLED
                job.message = "Cancelled by user."
            else:
                job.status = JobStatus.SUCCEEDED
                job.progress = 1.0
                job.message = "Done."
                job.result = result.to_dict() if hasattr(result, "to_dict") else result
        except Exception as exc:  # noqa: BLE001 - the job boundary must not leak
            if job.cancel_event.is_set():
                job.status = JobStatus.CANCELLED
                job.message = "Cancelled by user."
            else:
                logger.exception("job %s (%s) failed", job.id, job.kind)
                job.status = JobStatus.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = "Failed."
        finally:
            job.finished_at = time.time()
