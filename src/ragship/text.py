"""Small text helpers shared by chunking, embeddings, and extractive answers."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN.finditer(text)]


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])
