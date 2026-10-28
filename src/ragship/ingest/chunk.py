"""Overlapping character windows."""

from __future__ import annotations


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into windows of at most `size` characters.

    Adjacent windows share up to `overlap` characters. The start index always
    advances, so a large overlap cannot loop forever.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk overlap must be >= 0 and smaller than chunk size")
    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(start + size, length)
        if end < length:
            split_at = normalized.rfind(" ", start, end)
            if split_at >= start + size // 2:
                end = split_at
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks
