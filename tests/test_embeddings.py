"""Hash embeddings stay local. The OpenAI embedder is covered with a mock transport."""

from __future__ import annotations

import json
import math

import httpx
import pytest

from ragship.retrieval.embeddings import (
    HashEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
)


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def test_hash_is_deterministic_and_normalized():
    embedder = HashEmbedder(384)
    first = embedder.embed("Hash provider stays offline")
    second = embedder.embed("Hash provider stays offline")
    assert first == second
    assert embedder.embed_batch(["Hash provider stays offline"]) == [first]
    assert len(first) == 384
    norm = math.sqrt(sum(value * value for value in first))
    assert norm == pytest.approx(1.0)
    unrelated = embedder.embed("Kiln glaze stoneware biscuit")
    assert _cosine(first, second) > _cosine(first, unrelated)


def test_openai_embedder_preserves_order():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/embeddings"
        assert body["input"] == ["alpha", "beta"]
        assert body["dimensions"] == 2
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    embedder = OpenAIEmbedder(
        base_url="http://embed.test/v1",
        api_key="test-key",
        model="text-embedding-3-small",
        dimension=2,
        transport=httpx.MockTransport(handler),
    )
    assert embedder.embed_batch(["alpha", "beta"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_sentence_transformer_reports_missing_extra(monkeypatch):
    real_import = __import__

    def fake_import(name, global_ns=None, local_ns=None, fromlist=(), level=0):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("nope")
        return real_import(name, global_ns, local_ns, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        SentenceTransformerEmbedder("all-MiniLM-L6-v2")
