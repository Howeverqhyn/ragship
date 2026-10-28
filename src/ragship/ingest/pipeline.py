"""Chunk a document and replace its vectors in the store."""

from __future__ import annotations

from dataclasses import dataclass

from ragship.ingest.chunk import chunk_text
from ragship.ingest.loaders import DocumentError
from ragship.retrieval.embeddings import Embedder
from ragship.retrieval.store import VectorStore
from ragship.text import tokenize


@dataclass(frozen=True)
class IngestResult:
    source: str
    chunk_ids: list[str]

    @property
    def chunks(self) -> int:
        return len(self.chunk_ids)


def ingest_text(
    store: VectorStore,
    embedder: Embedder,
    text: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestResult:
    if not tokenize(text):
        raise DocumentError("document contains no indexable text")
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    if not chunks:
        raise DocumentError("document contains no text")
    vectors = embedder.embed_batch(chunks)
    chunk_ids = [f"{source}#{index}" for index in range(len(chunks))]
    store.replace_source(source, chunk_ids, vectors, chunks)
    return IngestResult(source=source, chunk_ids=chunk_ids)
