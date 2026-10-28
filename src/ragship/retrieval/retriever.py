"""Embed a question and return the nearest stored chunks."""

from __future__ import annotations

from dataclasses import dataclass

from ragship.retrieval.embeddings import Embedder
from ragship.retrieval.store import VectorStore
from ragship.text import tokenize


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    source: str
    score: float
    text: str


def retrieve(store: VectorStore, embedder: Embedder, query: str, top_k: int) -> list[Hit]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not tokenize(query):
        return []
    vector = embedder.embed(query)
    rows = store.search(vector, top_k)
    return [
        Hit(
            chunk_id=row["chunk_id"],
            source=row["source"],
            score=row["score"],
            text=row["text"],
        )
        for row in rows
    ]
