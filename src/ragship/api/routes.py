"""HTTP routes for health, ingest, and query."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request

from ragship import __version__
from ragship.api.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceHit,
)
from ragship.ingest.loaders import DocumentError, load_document, safe_source_name
from ragship.ingest.pipeline import ingest_text
from ragship.llm.client import LLMError
from ragship.retrieval.embeddings import EmbeddingError
from ragship.retrieval.retriever import retrieve
from ragship.retrieval.store import StoreError

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report process configuration. This check does not require an API key."""
    settings = request.app.state.settings
    reachable = request.app.state.store.ping()
    return HealthResponse(
        status="ok" if reachable else "degraded",
        qdrant=settings.qdrant_url,
        embedding_provider=settings.embedding_provider,
        llm_mode=settings.llm_mode,
        version=__version__,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request) -> IngestResponse:
    """Ingest a PDF, Markdown, or TXT document.

    Send multipart form data with a ``file`` field, or JSON
    ``{"text": "...", "source": "note.txt"}``.
    Re-ingesting the same source name replaces that source's chunks.
    """
    source, text = await _read_document(request)
    settings = request.app.state.settings
    try:
        result = ingest_text(
            store=request.app.state.store,
            embedder=request.app.state.embedder,
            text=text,
            source=source,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except DocumentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except StoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    logger.info("ingested source=%s chunks=%s", result.source, result.chunks)
    return IngestResponse(source=result.source, chunks=result.chunks, ids=result.chunk_ids)


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    """Retrieve the nearest chunks and answer from them."""
    settings = request.app.state.settings
    top_k = body.top_k if body.top_k is not None else settings.top_k
    try:
        hits = retrieve(
            store=request.app.state.store,
            embedder=request.app.state.embedder,
            query=body.query,
            top_k=top_k,
        )
        answer = request.app.state.llm.answer(body.query, [hit.text for hit in hits])
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except StoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    logger.info(
        "query chars=%s hits=%s mode=%s",
        len(body.query),
        len(hits),
        request.app.state.llm.mode,
    )
    return QueryResponse(
        answer=answer,
        mode=request.app.state.llm.mode,
        sources=[
            SourceHit(
                chunk_id=hit.chunk_id,
                source=hit.source,
                score=hit.score,
                text=hit.text,
            )
            for hit in hits
        ],
    )


async def _read_document(request: Request) -> tuple[str, str]:
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type == "application/json":
        return await _read_json(request)
    return await _read_form(request)


async def _read_json(request: Request) -> tuple[str, str]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="request body must be JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="JSON body must be an object")
    text = payload.get("text")
    source = payload.get("source", "inline.txt")
    if not isinstance(text, str):
        raise HTTPException(status_code=422, detail="text must be a string")
    if not isinstance(source, str):
        raise HTTPException(status_code=422, detail="source must be a string")
    _reject_large(text.encode("utf-8"))
    try:
        name = safe_source_name(source)
    except DocumentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    normalized = _normalize_text(text)
    if not normalized:
        raise HTTPException(status_code=400, detail="document contains no text")
    return name, normalized


async def _read_form(request: Request) -> tuple[str, str]:
    try:
        form = await request.form()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="expected a file upload or JSON body") from exc
    upload = form.get("file")
    if upload is not None and hasattr(upload, "read"):
        data = await upload.read()
        if not isinstance(data, (bytes, bytearray)):
            raise HTTPException(status_code=400, detail="file upload was not binary")
        raw = bytes(data)
        _reject_large(raw)
        raw_name = form.get("source") or getattr(upload, "filename", None) or "upload.txt"
        if not isinstance(raw_name, str):
            raise HTTPException(status_code=422, detail="source must be a string")
        try:
            text = load_document(raw_name, raw)
            name = safe_source_name(raw_name)
        except DocumentError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        return name, text

    raw_text = form.get("text")
    if not isinstance(raw_text, str):
        raise HTTPException(status_code=400, detail="provide a file or text")
    raw_name = form.get("source") or "inline.txt"
    if not isinstance(raw_name, str):
        raise HTTPException(status_code=422, detail="source must be a string")
    _reject_large(raw_text.encode("utf-8"))
    try:
        name = safe_source_name(raw_name)
    except DocumentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    normalized = _normalize_text(raw_text)
    if not normalized:
        raise HTTPException(status_code=400, detail="document contains no text")
    return name, normalized


def _reject_large(data: bytes) -> None:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="document exceeds 20 MB limit")


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()
