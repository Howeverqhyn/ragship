"""In-process Qdrant ranks the chunk that shares the query's words."""

from __future__ import annotations

from ragship.config import load_settings
from ragship.ingest.pipeline import ingest_text
from ragship.retrieval.embeddings import build_embedder
from ragship.retrieval.retriever import retrieve
from ragship.retrieval.store import VectorStore


def test_memory_store_ranks_overlapping_text():
    settings = load_settings()
    settings.qdrant_url = "memory"
    settings.collection = "rank-test"
    embedder = build_embedder(settings)
    settings.vector_size = embedder.dimension
    store = VectorStore(settings)
    ingest_text(
        store,
        embedder,
        "Qdrant memory mode keeps vectors in process.",
        "q.txt",
        800,
        120,
    )
    ingest_text(
        store,
        embedder,
        "Sourdough needs a long cold proof in the fridge.",
        "b.txt",
        800,
        120,
    )
    hits = retrieve(store, embedder, "How does Qdrant memory mode store vectors?", 2)
    assert hits[0].source == "q.txt"
    assert "Qdrant" in hits[0].text
