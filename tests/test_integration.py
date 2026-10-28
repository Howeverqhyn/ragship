"""Live Qdrant round trip. Skipped unless RAGSHIP_INTEGRATION=1."""

from __future__ import annotations

import os
import uuid

import pytest

from ragship.config import load_settings
from ragship.ingest.pipeline import ingest_text
from ragship.retrieval.embeddings import build_embedder
from ragship.retrieval.retriever import retrieve
from ragship.retrieval.store import VectorStore, is_memory_url

pytestmark = pytest.mark.skipif(
    os.environ.get("RAGSHIP_INTEGRATION") != "1",
    reason="set RAGSHIP_INTEGRATION=1 to run live Qdrant tests",
)


def test_live_qdrant_roundtrip():
    settings = load_settings()
    if is_memory_url(settings.qdrant_url):
        pytest.fail("RAGSHIP_INTEGRATION=1 requires QDRANT_URL to point at a Qdrant server")
    settings.collection = f"ragship-it-{uuid.uuid4().hex[:8]}"
    embedder = build_embedder(settings)
    settings.vector_size = embedder.dimension
    store = VectorStore(settings)
    try:
        ingest_text(
            store,
            embedder,
            "Integration ragship token ORCHIDVECTOR lives in Qdrant.",
            "integration.txt",
            settings.chunk_size,
            settings.chunk_overlap,
        )
        hits = retrieve(store, embedder, "Where does ORCHIDVECTOR live?", top_k=3)
        assert hits
        assert hits[0].source == "integration.txt"
        assert "ORCHIDVECTOR" in hits[0].text
    finally:
        store.drop()
