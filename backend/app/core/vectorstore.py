"""Persistence for embeddings (Repository pattern over ChromaDB).

The rest of the application talks to :class:`VectorRepository` in terms of
domain objects (``VectorRecord``, ``RetrievedChunk``) and never imports
``chromadb``. Swapping in FAISS, Qdrant or pgvector later means writing one
new class that satisfies the same five methods.

One Chroma collection per indexed root keeps searches naturally scoped and
makes "forget this folder" a single ``delete_collection`` call.

Distance metric is cosine. Ollama embedding models are not unit-normalised, so
Euclidean distance would conflate document *length* with document *meaning*;
cosine compares direction only.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import chromadb
from chromadb.config import Settings as ChromaClientSettings

if TYPE_CHECKING:  # the concrete class lives on an internal path that moves
    from chromadb.api.models.Collection import Collection

from app.config import Settings
from app.core.paths import collection_name

logger = logging.getLogger(__name__)

MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]

KIND_CHUNK = "chunk"
KIND_FOLDER_SUMMARY = "folder_summary"


@dataclass(slots=True)
class VectorRecord:
    """One embeddable unit: a document chunk or a folder summary."""

    id: str
    document: str
    embedding: list[float]
    metadata: Metadata


@dataclass(slots=True)
class RetrievedChunk:
    id: str
    document: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rel_path(self) -> str:
        return str(self.metadata.get("rel_path", ""))

    @property
    def abs_path(self) -> str:
        return str(self.metadata.get("abs_path", ""))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["relPath"] = self.rel_path
        data["absPath"] = self.abs_path
        return data


def sanitize_metadata(raw: dict[str, Any]) -> Metadata:
    """Coerce a dict into Chroma's scalar-only metadata contract.

    Chroma rejects ``None`` and nested/complex values outright, so lists are
    joined and everything else is stringified rather than raising mid-index.
    """
    clean: Metadata = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, bool | int | float | str):
            clean[key] = value
        elif isinstance(value, list | tuple | set):
            joined = ",".join(str(v) for v in value if v is not None)
            if joined:
                clean[key] = joined
        else:
            clean[key] = str(value)
    return clean


class VectorRepository:
    """Thread-safe facade over a persistent Chroma client."""

    _client_lock = threading.Lock()
    _client: chromadb.ClientAPI | None = None

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # -------------------------------------------------------------- plumbing
    @classmethod
    def _get_client(cls, settings: Settings) -> chromadb.ClientAPI:
        """Single shared client.

        Chroma's persistent backend holds an exclusive lock on its SQLite
        file; two clients on one directory in the same process is a
        documented way to corrupt state. Double-checked locking keeps the
        happy path lock-free after the first call.
        """
        if cls._client is None:
            with cls._client_lock:
                if cls._client is None:
                    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
                    cls._client = chromadb.PersistentClient(
                        path=str(settings.chroma_dir),
                        settings=ChromaClientSettings(
                            anonymized_telemetry=False,
                            allow_reset=True,
                        ),
                    )
        return cls._client

    def _collection(self, root: Path) -> "Collection":
        """Fetch (or create) this root's collection.

        ``embedding_function=None`` is the important argument: we always pass
        pre-computed vectors, and leaving it unset makes Chroma attach its
        bundled ONNX MiniLM model, which downloads weights on first use. That
        would silently break the "everything stays local" guarantee.

        The kwarg variants exist because the collection-configuration API has
        been renamed across Chroma releases; each is tried in order of
        preference and the last one always works.
        """
        client = self._get_client(self._settings)
        name = collection_name(root)
        metadata = {"hnsw:space": "cosine", "source_root": str(root)}

        variants: list[dict[str, Any]] = [
            {"embedding_function": None, "metadata": metadata},
            {"metadata": metadata},
            {"embedding_function": None},
            {},
        ]
        last_error: Exception | None = None
        for kwargs in variants:
            try:
                return client.get_or_create_collection(name=name, **kwargs)
            except Exception as exc:  # noqa: BLE001 - probing for a supported signature
                last_error = exc
                logger.debug("collection kwargs %s rejected: %s", list(kwargs), exc)
        raise RuntimeError(f"Could not open Chroma collection {name}: {last_error}")

    # ----------------------------------------------------------------- write
    def upsert(self, root: Path, records: Iterable[VectorRecord]) -> int:
        """Insert or replace records, batched to bound peak memory.

        ``upsert`` (not ``add``) makes re-indexing idempotent: chunk ids are
        deterministic, so re-running over an unchanged file is a no-op rather
        than a duplicate.
        """
        collection = self._collection(root)
        batch_size = max(1, self._settings.vector_upsert_batch_size)
        buffer: list[VectorRecord] = []
        written = 0

        def flush() -> None:
            nonlocal written
            if not buffer:
                return
            collection.upsert(
                ids=[r.id for r in buffer],
                documents=[r.document for r in buffer],
                embeddings=[r.embedding for r in buffer],
                metadatas=[r.metadata for r in buffer],
            )
            written += len(buffer)
            buffer.clear()

        for record in records:
            buffer.append(record)
            if len(buffer) >= batch_size:
                flush()
        flush()
        return written

    def delete_by_rel_paths(self, root: Path, rel_paths: Iterable[str]) -> int:
        """Remove every vector belonging to the given files.

        Chroma's ``$in`` operator is chunked because the filter is compiled
        into a SQL ``IN (...)`` clause, and SQLite caps bound parameters
        (999 on many builds).
        """
        paths = [p for p in rel_paths if p]
        if not paths:
            return 0
        collection = self._collection(root)
        window = 400
        for start in range(0, len(paths), window):
            collection.delete(where={"rel_path": {"$in": paths[start : start + window]}})
        return len(paths)

    def delete_kind(self, root: Path, kind: str) -> None:
        self._collection(root).delete(where={"kind": kind})

    def drop(self, root: Path) -> None:
        try:
            self._get_client(self._settings).delete_collection(collection_name(root))
        except Exception:  # noqa: BLE001 - deleting a non-existent collection is fine
            logger.debug("collection for %s already absent", root, exc_info=True)

    # ------------------------------------------------------------------ read
    def count(self, root: Path) -> int:
        try:
            return self._collection(root).count()
        except Exception:  # noqa: BLE001
            return 0

    def query(
        self,
        root: Path,
        embedding: list[float],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Approximate nearest-neighbour search.

        Chroma indexes with HNSW, a navigable small-world graph: query cost is
        ~O(log N) hops instead of the O(N) of a brute-force scan, at the price
        of approximate (not guaranteed exact) neighbours.
        """
        collection = self._collection(root)
        if collection.count() == 0:
            return []

        try:
            raw = collection.query(
                query_embeddings=[embedding],
                n_results=max(1, min(n_results, collection.count())),
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:  # noqa: BLE001 - malformed filter shouldn't 500 the API
            logger.warning("vector query failed for %s", root, exc_info=True)
            return []

        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            results.append(
                RetrievedChunk(
                    id=str(chunk_id),
                    document=str(documents[index]) if index < len(documents) else "",
                    # Chroma returns cosine *distance*; similarity is 1 - d,
                    # clamped because floating-point error can nudge it below 0.
                    score=max(0.0, min(1.0, 1.0 - distance)),
                    metadata=dict(metadatas[index] or {}) if index < len(metadatas) else {},
                )
            )
        return results

    def get_folder_summaries(self, root: Path) -> dict[str, str]:
        """All stored folder summaries as ``{rel_path: summary}``."""
        try:
            raw = self._collection(root).get(
                where={"kind": KIND_FOLDER_SUMMARY},
                include=["documents", "metadatas"],
            )
        except Exception:  # noqa: BLE001
            return {}
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        return {
            str((metadatas[i] or {}).get("rel_path", "")): str(documents[i])
            for i in range(min(len(documents), len(metadatas)))
        }
