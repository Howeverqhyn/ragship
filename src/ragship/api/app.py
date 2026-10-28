"""FastAPI application factory and the serve command."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ragship import __version__
from ragship.api.routes import router
from ragship.config import Settings, load_settings
from ragship.llm.client import build_llm
from ragship.retrieval.embeddings import build_embedder
from ragship.retrieval.store import VectorStore

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or load_settings()
    embedder = build_embedder(resolved)
    resolved.vector_size = embedder.dimension
    store = VectorStore(resolved)
    llm = build_llm(resolved)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info(
            "ragship %s qdrant=%s embedding=%s llm=%s",
            __version__,
            resolved.qdrant_url,
            resolved.embedding_provider,
            resolved.llm_mode,
        )
        yield

    app = FastAPI(
        title="ragship",
        version=__version__,
        summary="Retrieval-augmented generation over documents you ingest.",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.embedder = embedder
    app.state.store = store
    app.state.llm = llm
    app.include_router(router)
    return app


def serve(host: str | None = None, port: int | None = None) -> None:
    settings = load_settings()
    if host:
        settings.host = host
    if port is not None:
        if port < 1 or port > 65535:
            print("ragship: port must be between 1 and 65535", file=sys.stderr)
            raise SystemExit(2)
        settings.port = port
    try:
        app = create_app(settings)
    except (ValueError, RuntimeError) as exc:
        print(f"ragship: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(app, host=settings.host, port=settings.port)
