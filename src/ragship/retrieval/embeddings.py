"""Offline hash embeddings, plus optional neural and OpenAI embedders."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import httpx

from ragship import __version__
from ragship.config import Settings
from ragship.text import tokenize


class EmbeddingError(RuntimeError):
    """The embedding provider could not produce a vector."""


class Embedder(Protocol):
    dimension: int

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Signed feature hashing. Deterministic, local, and model-free.

    Each token contributes to two buckets. The vector is L2-normalized so
    Qdrant cosine search can rank lexical overlap.
    """

    def __init__(self, dimension: int) -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(text, self.dimension) for text in texts]


class SentenceTransformerEmbedder:
    """Optional neural embedder. Import is deferred so the default path stays light."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                'Install it with: pip install -e ".[sentence-transformers]"'
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


class OpenAIEmbedder:
    """POST {OPENAI_BASE_URL}/embeddings. Used when EMBEDDING_PROVIDER=openai."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimension: int,
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("OPENAI_BASE_URL is required for embedding provider 'openai'")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.timeout = timeout
        self.transport = transport

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict = {"model": self.model, "input": texts}
        if "text-embedding-3" in self.model:
            payload["dimensions"] = self.dimension
        body = _post_json(
            url=f"{self.base_url}/embeddings",
            payload=payload,
            api_key=self.api_key,
            timeout=self.timeout,
            transport=self.transport,
        )
        data = body.get("data")
        if not isinstance(data, list):
            raise EmbeddingError("embedding response was missing data")
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors: list[list[float]] = []
        for item in ordered:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise EmbeddingError("embedding response was missing a vector")
            if len(embedding) != self.dimension:
                raise EmbeddingError(
                    f"embedding model returned dimension {len(embedding)}, "
                    f"expected {self.dimension}. Set EMBEDDING_DIM to match the model."
                )
            vectors.append([float(value) for value in embedding])
        if len(vectors) != len(texts):
            raise EmbeddingError("embedding response did not include every input")
        return vectors


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider
    if provider == "hash":
        return HashEmbedder(settings.vector_size)
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    if provider == "openai":
        return OpenAIEmbedder(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimension=settings.vector_size,
        )
    raise ValueError(f"unknown embedding provider: {provider}")


def _hash_embed(text: str, dimension: int) -> list[float]:
    features = _features(text)
    vector = [0.0] * dimension
    if not features:
        vector[0] = 1.0
        return vector
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        for offset in (0, 8):
            bucket = int.from_bytes(digest[offset : offset + 4], "little") % dimension
            sign = 1.0 if digest[offset + 4] % 2 == 0 else -1.0
            vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


def _features(text: str) -> list[str]:
    tokens = tokenize(text)
    bigrams = [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
    return tokens + bigrams


def _post_json(
    *,
    url: str,
    payload: dict,
    api_key: str,
    timeout: float,
    transport: httpx.BaseTransport | None,
) -> dict:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"ragship/{__version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise EmbeddingError(f"embedding request failed: {exc}") from exc
    if response.status_code >= 400:
        detail = response.text.strip().replace("\n", " ")[:300]
        raise EmbeddingError(f"embedding request failed with HTTP {response.status_code}: {detail}")
    try:
        body = response.json()
    except ValueError as exc:
        raise EmbeddingError("embedding response was not JSON") from exc
    if not isinstance(body, dict):
        raise EmbeddingError("embedding response was not a JSON object")
    return body
