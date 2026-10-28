"""Set recall at a fixed cutoff."""

from __future__ import annotations

from collections.abc import Collection, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str]) -> float:
    """|relevant ∩ retrieved| / |relevant|.

    Document-level: pass source ids, not chunk ids. An empty gold set scores 0.
    """
    relevant_set = {item for item in relevant if item}
    if not relevant_set:
        return 0.0
    found = relevant_set.intersection(retrieved)
    return len(found) / len(relevant_set)


def mean_recall(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
