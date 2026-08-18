"""Ollama-backed chat and embedding providers.

Both handles are lazily-constructed process singletons. That matters for more
than tidiness: ``OllamaEmbeddings`` and ``ChatOllama`` each own an HTTP client
with a connection pool, so rebuilding them per request would mean a fresh TCP
(and keep-alive) handshake on every call.

Nothing here talks to a hosted API. Every token and every vector stays on the
machine running Ollama, which is the entire privacy premise of the product.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import Settings, get_settings
from app.core.textutils import message_text

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when the local Ollama daemon cannot serve a request."""


@dataclass(frozen=True, slots=True)
class OllamaStatus:
    reachable: bool
    base_url: str
    models: tuple[str, ...]
    chat_model_ready: bool
    embedding_model_ready: bool
    detail: str = ""


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOllama:
    settings = get_settings()
    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_ctx=settings.llm_num_ctx,
        client_kwargs={"timeout": settings.llm_request_timeout_s},
    )


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    settings = get_settings()
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def reset_providers() -> None:
    """Drop cached clients (used after a settings change or in tests)."""
    get_chat_model.cache_clear()
    get_embeddings.cache_clear()


# --------------------------------------------------------------------- health
def _model_matches(available: str, wanted: str) -> bool:
    """Ollama reports ``llama3.1:8b``; users often configure ``llama3.1``."""
    return available == wanted or available.split(":", 1)[0] == wanted.split(":", 1)[0]


async def check_ollama(settings: Settings | None = None) -> OllamaStatus:
    settings = settings or get_settings()
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return OllamaStatus(
            reachable=False,
            base_url=settings.ollama_base_url,
            models=(),
            chat_model_ready=False,
            embedding_model_ready=False,
            detail=f"{type(exc).__name__}: {exc}",
        )

    names = tuple(str(m.get("name", "")) for m in payload.get("models", []))
    return OllamaStatus(
        reachable=True,
        base_url=settings.ollama_base_url,
        models=names,
        chat_model_ready=any(_model_matches(n, settings.chat_model) for n in names),
        embedding_model_ready=any(
            _model_matches(n, settings.embedding_model) for n in names
        ),
    )


# ------------------------------------------------------------------ shortcuts
def build_messages(system: str, user: str) -> list[BaseMessage]:
    return [SystemMessage(content=system), HumanMessage(content=user)]


def complete(system: str, user: str) -> str:
    """Synchronous single-turn completion, for use inside worker threads."""
    try:
        response = get_chat_model().invoke(build_messages(system, user))
    except Exception as exc:  # noqa: BLE001 - surface a typed error upward
        raise LLMUnavailableError(str(exc)) from exc
    return message_text(response).strip()


async def acomplete(system: str, user: str) -> str:
    try:
        response = await get_chat_model().ainvoke(build_messages(system, user))
    except Exception as exc:  # noqa: BLE001
        raise LLMUnavailableError(str(exc)) from exc
    return message_text(response).strip()


def embed_documents(texts: list[str], batch_size: int) -> list[list[float]]:
    """Embed in fixed-size batches.

    One request per batch amortises HTTP overhead while keeping any single
    payload small enough to avoid server-side timeouts on a cold model.
    """
    if not texts:
        return []
    embedder = get_embeddings()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), max(1, batch_size)):
        window = texts[start : start + batch_size]
        try:
            vectors.extend(embedder.embed_documents(window))
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailableError(
                f"Embedding failed (is '{get_settings().embedding_model}' pulled?): {exc}"
            ) from exc
    return vectors


def embed_query(text: str) -> list[float]:
    try:
        return get_embeddings().embed_query(text)
    except Exception as exc:  # noqa: BLE001
        raise LLMUnavailableError(str(exc)) from exc
