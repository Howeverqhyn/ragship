"""Qdrant collection wrapper. QDRANT_URL=memory stays inside the process."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from ragship.config import Settings

logger = logging.getLogger(__name__)

_NAMESPACE = uuid.UUID("8f1b6c2e-9a4d-4e1a-9c3b-6d0e5f7a8b21")
_RETRYABLE = (UnexpectedResponse, httpx.HTTPError, OSError, TimeoutError)


class StoreError(RuntimeError):
    """A remote Qdrant call failed."""


def is_memory_url(url: str) -> bool:
    return url.strip().lower() in {"memory", ":memory:", "mem"}


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self._url = settings.qdrant_url
        self._collection = settings.collection
        self._vector_size = settings.vector_size
        self._memory = is_memory_url(self._url)
        self._lock = threading.Lock()
        self._ready = False
        if self._memory:
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(url=self._url, timeout=10, prefer_grpc=False)
        self._connect()

    def replace_source(
        self,
        source: str,
        chunk_ids: list[str],
        vectors: list[list[float]],
        chunks: list[str],
    ) -> None:
        if not (len(chunk_ids) == len(vectors) == len(chunks)):
            raise ValueError("chunk ids, vectors, and texts must be the same length")
        points = [
            PointStruct(
                id=_point_id(chunk_id),
                vector=vector,
                payload={
                    "text": chunk,
                    "source": source,
                    "chunk_id": chunk_id,
                    "index": index,
                },
            )
            for index, (chunk_id, vector, chunk) in enumerate(zip(chunk_ids, vectors, chunks))
        ]
        with self._lock:
            self._ensure_collection()
            self._delete_source(source)
            if points:
                self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def search(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        with self._lock:
            self._ensure_collection()
            response = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
        hits: list[dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            text = payload.get("text")
            source = payload.get("source")
            chunk_id = payload.get("chunk_id")
            if not isinstance(text, str) or not isinstance(source, str):
                continue
            hits.append(
                {
                    "chunk_id": chunk_id if isinstance(chunk_id, str) else str(point.id),
                    "source": source,
                    "score": float(point.score),
                    "text": text,
                }
            )
        return hits

    def ping(self) -> bool:
        try:
            with self._lock:
                self._client.get_collections()
        except Exception:
            logger.warning("qdrant ping failed", exc_info=True)
            return False
        return True

    def drop(self) -> None:
        with self._lock:
            if self._client.collection_exists(self._collection):
                self._client.delete_collection(self._collection)
            self._ready = False

    def _connect(self) -> None:
        if self._memory:
            with self._lock:
                self._ensure_collection()
            return
        delay = 0.5
        last: Exception | None = None
        for _ in range(8):
            try:
                with self._lock:
                    self._ensure_collection()
                return
            except _RETRYABLE as exc:
                last = exc
                logger.info("waiting for Qdrant at %s", self._url)
                time.sleep(delay)
                delay = min(delay * 1.7, 4)
        raise StoreError(f"could not reach Qdrant at {self._url}") from last

    def _ensure_collection(self) -> None:
        if self._ready:
            return
        exists = self._client.collection_exists(self._collection)
        if exists:
            info = self._client.get_collection(self._collection)
            size = _configured_vector_size(info)
            if size is not None and size != self._vector_size:
                logger.info(
                    "recreating collection %s because vector size changed from %s to %s",
                    self._collection,
                    size,
                    self._vector_size,
                )
                self._client.delete_collection(self._collection)
                exists = False
        if not exists:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )
            if not self._memory:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name="source",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
        self._ready = True

    def _delete_source(self, source: str) -> None:
        selector = FilterSelector(
            filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
        )
        self._client.delete(collection_name=self._collection, points_selector=selector)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def _configured_vector_size(info: object) -> int | None:
    config = getattr(info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    size = getattr(vectors, "size", None)
    if isinstance(size, int):
        return size
    if isinstance(vectors, dict):
        first = next(iter(vectors.values()), None)
        named = getattr(first, "size", None)
        if isinstance(named, int):
            return named
    return None
