"""ASGI application factory."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.core.llm import check_ollama
from app.core.paths import PathSecurityError
from app.services.container import get_container

logger = logging.getLogger("ais")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Chroma and httpx are chatty at INFO and drown out our own progress logs.
    for noisy in ("chromadb", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the container on boot and report Ollama readiness once.

    Building the container eagerly means the first user request does not pay
    for opening the Chroma database, and a misconfigured model shows up in the
    server log at startup rather than as a confusing 503 later.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    container = get_container()
    # Watcher callbacks fire on watchdog's own worker threads; binding the
    # loop here lets them hop back onto it via run_coroutine_threadsafe.
    container.watchers.bind_loop(asyncio.get_running_loop())

    ollama = await check_ollama(settings)
    if not ollama.reachable:
        logger.warning(
            "Ollama unreachable at %s (%s). Start it with `ollama serve`.",
            ollama.base_url,
            ollama.detail,
        )
    else:
        if not ollama.chat_model_ready:
            logger.warning("Chat model '%s' not pulled. Run: ollama pull %s",
                           settings.chat_model, settings.chat_model)
        if not ollama.embedding_model_ready:
            logger.warning("Embedding model '%s' not pulled. Run: ollama pull %s",
                           settings.embedding_model, settings.embedding_model)
        logger.info("Ollama ready at %s", ollama.base_url)

    logger.info("Data directory: %s", settings.data_dir)
    yield
    logger.info("Shutting down.")
    await asyncio.to_thread(container.watchers.stop_all)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AI-Powered Internal Search",
        version="0.1.0",
        summary="Local-first semantic search, summarisation and auto-filing for folders.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(PathSecurityError)
    async def _path_error(_request: Request, exc: PathSecurityError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)}
        )

    @app.middleware("http")
    async def _log_unmatched_probes(request: Request, call_next):
        """Diagnostic: identify what is probing unknown paths like /json/version.

        Uvicorn's access log only records method/path/status, which isn't
        enough to tell a stray CDP/devtools probe apart from a real client
        bug. Logging the culprit's headers on a miss costs nothing on the
        happy path. Remove once the source is confirmed.
        """
        response = await call_next(request)
        if response.status_code == status.HTTP_404_NOT_FOUND:
            headers = request.headers
            logger.warning(
                "404 probe: %s %s | UA=%r Referer=%r Origin=%r client=%s",
                request.method,
                request.url.path,
                headers.get("user-agent"),
                headers.get("referer"),
                headers.get("origin"),
                request.client,
            )
        return response

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def index() -> dict[str, str]:
        return {"service": "ai-internal-search", "docs": "/docs"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    _settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=_settings.host,
        port=_settings.port,
        reload=True,
        log_level=_settings.log_level.lower(),
    )
