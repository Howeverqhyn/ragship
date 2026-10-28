"""Ingest a corpus into in-process Qdrant and score labeled queries."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from ragship.config import Settings, load_settings, resolve_from_root
from ragship.eval.metrics import mean_recall, recall_at_k
from ragship.ingest.loaders import is_supported_name, load_document
from ragship.ingest.pipeline import ingest_text
from ragship.retrieval.embeddings import Embedder, build_embedder
from ragship.retrieval.retriever import retrieve
from ragship.retrieval.store import VectorStore


@dataclass(frozen=True)
class LabeledExample:
    query: str
    relevant_ids: list[str]


@dataclass(frozen=True)
class ExampleResult:
    query: str
    recall: float
    relevant_ids: list[str]
    retrieved_sources: list[str]

    @property
    def hit(self) -> bool:
        return self.recall >= 1.0


@dataclass(frozen=True)
class EvalReport:
    k: int
    dataset: str
    recall: float
    results: list[ExampleResult]


def run_eval(
    *,
    k: int | None = None,
    dataset: str | Path | None = None,
    corpus: str | Path | None = None,
    settings: Settings | None = None,
) -> EvalReport:
    """Score Recall@k. Always uses an in-process Qdrant, ignoring QDRANT_URL."""
    base = replace(settings or load_settings())
    base.qdrant_url = "memory"
    base.collection = f"ragship-eval-{uuid.uuid4().hex[:8]}"
    if dataset is not None:
        base.eval_dataset = resolve_from_root(base.root, dataset)
    if corpus is not None:
        base.eval_corpus = resolve_from_root(base.root, corpus)
    cutoff = k if k is not None else base.eval_k
    if cutoff < 1:
        raise ValueError("k must be >= 1")

    examples = load_labeled(base.eval_dataset)
    embedder = build_embedder(base)
    base.vector_size = embedder.dimension
    store = VectorStore(base)
    _ingest_corpus(store, embedder, base)

    results: list[ExampleResult] = []
    for example in examples:
        hits = retrieve(store, embedder, example.query, cutoff)
        sources = _unique_sources(hit.source for hit in hits)
        score = recall_at_k(sources, example.relevant_ids)
        results.append(
            ExampleResult(
                query=example.query,
                recall=score,
                relevant_ids=list(example.relevant_ids),
                retrieved_sources=sources,
            )
        )
    return EvalReport(
        k=cutoff,
        dataset=str(base.eval_dataset),
        recall=mean_recall([item.recall for item in results]),
        results=results,
    )


def format_report(report: EvalReport) -> str:
    lines = [
        f"dataset: {report.dataset}",
        f"k: {report.k}",
        f"examples: {len(report.results)}",
        f"recall@{report.k}: {report.recall:.3f}",
    ]
    for result in report.results:
        if result.recall < 1.0:
            lines.append(
                "MISS "
                f"query={result.query!r} "
                f"relevant={result.relevant_ids} "
                f"retrieved={result.retrieved_sources}"
            )
    return "\n".join(lines)


def load_labeled(path: Path) -> list[LabeledExample]:
    if not path.is_file():
        raise ValueError(f"labeled dataset not found: {path}")
    examples: list[LabeledExample] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_no}: each line must be an object")
        query = raw.get("query")
        relevant = raw.get("relevant_ids")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{path}:{line_no}: query is required")
        if (
            not isinstance(relevant, list)
            or not relevant
            or not all(isinstance(item, str) for item in relevant)
        ):
            raise ValueError(f"{path}:{line_no}: relevant_ids must be a non-empty list of strings")
        examples.append(LabeledExample(query=query.strip(), relevant_ids=list(relevant)))
    if not examples:
        raise ValueError(f"no labeled examples in {path}")
    return examples


def _ingest_corpus(store: VectorStore, embedder: Embedder, settings: Settings) -> None:
    corpus = settings.eval_corpus
    if not corpus.is_dir():
        raise ValueError(f"eval corpus not found: {corpus}")
    files = sorted(
        path for path in corpus.iterdir() if path.is_file() and is_supported_name(path.name)
    )
    if not files:
        raise ValueError(f"eval corpus is empty: {corpus}")
    for path in files:
        text = load_document(path.name, path.read_bytes())
        ingest_text(
            store,
            embedder,
            text,
            path.name,
            settings.chunk_size,
            settings.chunk_overlap,
        )


def _unique_sources(sources: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for source in sources:
        if source not in seen:
            seen.add(source)
            ordered.append(source)
    return ordered
