"""Recall@k on the labeled demo set, using in-process Qdrant."""

from __future__ import annotations

from ragship.eval.harness import format_report, run_eval
from ragship.eval.metrics import mean_recall, recall_at_k


def test_recall_at_k_is_set_overlap():
    assert recall_at_k(["a.md", "b.md"], {"b.md"}) == 1.0
    assert recall_at_k(["a.md"], {"b.md"}) == 0.0
    assert recall_at_k(["a.md", "c.md"], ["a.md", "b.md"]) == 0.5
    assert recall_at_k(["a.md"], []) == 0.0
    assert mean_recall([1.0, 0.0, 1.0, 1.0]) == 0.75
    assert mean_recall([]) == 0.0


def test_demo_recall_is_perfect(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")
    report = run_eval()
    assert report.k == 5
    assert len(report.results) == 6
    assert report.recall == 1.0, format_report(report)
    assert "recall@5: 1.000" in format_report(report)
