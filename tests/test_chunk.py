"""Chunk windows cover the source text and overlap on the boundary."""

from __future__ import annotations

import pytest

from ragship.ingest.chunk import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("hello", 100, 10) == ["hello"]
    assert chunk_text("   \n", 100, 10) == []


def test_overlap_repeats_the_boundary():
    text = "0123456789" * 20
    chunks = chunk_text(text, size=50, overlap=10)
    assert len(chunks) > 2
    assert chunks[0][-10:] == chunks[1][:10]


def test_chunks_cover_every_word():
    text = " ".join(f"word{index:04d}" for index in range(200))
    chunks = chunk_text(text, size=80, overlap=20)
    assert len(chunks) > 1
    for word in text.split():
        assert any(word in chunk for chunk in chunks)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("hello", 10, 10)
