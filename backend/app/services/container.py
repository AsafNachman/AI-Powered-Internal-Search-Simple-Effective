"""Composition root.

Every collaborator is constructed exactly once, here, and reached through
FastAPI's ``Depends``. Routers therefore declare *what* they need instead of
importing and instantiating it, which keeps them free of wiring logic and
makes each one testable with a stub container.

This is manual dependency injection -- no framework required, because Python's
default arguments plus one module-level singleton already give us everything a
DI container would.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config import Settings, get_settings
from app.core.cleanup import CleanupAnalyzer
from app.core.extractors import ExtractorRegistry
from app.core.filing_agent import FilingAgent
from app.core.indexer import IndexingService
from app.core.manifest import ManifestStore
from app.core.scanner import DirectoryScanner
from app.core.vectorstore import VectorRepository
from app.services.jobs import JobManager
from app.services.rag import RagService
from app.services.watch import WatchManager


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    registry: ExtractorRegistry
    repository: VectorRepository
    manifests: ManifestStore
    scanner: DirectoryScanner
    indexer: IndexingService
    rag: RagService
    cleanup: CleanupAnalyzer
    filing: FilingAgent
    jobs: JobManager
    watchers: WatchManager


@lru_cache(maxsize=1)
def get_container() -> Container:
    settings = get_settings()
    registry = ExtractorRegistry()
    repository = VectorRepository(settings)
    manifests = ManifestStore(settings)
    indexer = IndexingService(settings, repository, manifests, registry)
    jobs = JobManager()
    return Container(
        settings=settings,
        registry=registry,
        repository=repository,
        manifests=manifests,
        scanner=DirectoryScanner(settings),
        indexer=indexer,
        rag=RagService(settings, repository),
        cleanup=CleanupAnalyzer(settings),
        filing=FilingAgent(settings, repository, registry),
        jobs=jobs,
        watchers=WatchManager(settings, indexer, jobs),
    )
