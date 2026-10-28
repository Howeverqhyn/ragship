"""Load YAML defaults and let environment variables override them."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_EMBEDDING_PROVIDERS = {"hash", "sentence-transformers", "openai"}
_LLM_MODES = {"extractive", "openai"}


@dataclass
class Settings:
    host: str
    port: int
    qdrant_url: str
    collection: str
    vector_size: int
    embedding_provider: str
    embedding_model: str
    openai_base_url: str
    openai_api_key: str = field(repr=False)
    model: str
    llm_mode: str
    temperature: float
    chunk_size: int
    chunk_overlap: int
    top_k: int
    eval_k: int
    min_recall: float
    eval_dataset: Path
    eval_corpus: Path
    root: Path


def project_root() -> Path:
    """Directory that contains configs/default.yaml, else the process cwd."""
    override = _env("RAGSHIP_ROOT")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    packaged = here.parents[2]
    if (packaged / "configs" / "default.yaml").is_file():
        return packaged
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "configs" / "default.yaml").is_file():
            return candidate
    return cwd


def load_settings() -> Settings:
    root = project_root()
    config_override = _env("RAGSHIP_CONFIG")
    config_path = Path(config_override) if config_override else root / "configs" / "default.yaml"
    if not config_path.is_absolute():
        config_path = root / config_path
    data = _read_yaml(config_path)

    server = _section(data, "server")
    qdrant = _section(data, "qdrant")
    embedding = _section(data, "embedding")
    llm = _section(data, "llm")
    ingest = _section(data, "ingest")
    retrieval = _section(data, "retrieval")
    evaluation = _section(data, "eval")

    provider = (
        _env("EMBEDDING_PROVIDER") or str(embedding.get("provider") or "hash")
    ).lower().replace("_", "-")
    if provider not in _EMBEDDING_PROVIDERS:
        allowed = ", ".join(sorted(_EMBEDDING_PROVIDERS))
        raise ValueError(f"EMBEDDING_PROVIDER must be one of: {allowed}")

    embedding_model = _env("EMBEDDING_MODEL") or str(embedding.get("model") or "")
    embedding_model = _default_embedding_model(provider, embedding_model)

    openai_base_url = _env("OPENAI_BASE_URL") or str(llm.get("base_url") or "")
    explicit_mode = _env("LLM_PROVIDER")
    if explicit_mode:
        llm_mode = explicit_mode.lower()
    elif openai_base_url:
        llm_mode = "openai"
    else:
        llm_mode = "extractive"
    if llm_mode not in _LLM_MODES:
        raise ValueError("LLM_PROVIDER must be 'extractive' or 'openai'")

    settings = Settings(
        host=_env("HOST") or str(server.get("host") or "0.0.0.0"),
        port=_env_int("PORT", int(server.get("port", 8091))),
        qdrant_url=_env("QDRANT_URL") or str(qdrant.get("url") or "memory"),
        collection=_env("QDRANT_COLLECTION") or str(qdrant.get("collection") or "ragship"),
        vector_size=_env_int("EMBEDDING_DIM", int(embedding.get("dimension", 384))),
        embedding_provider=provider,
        embedding_model=embedding_model,
        openai_base_url=openai_base_url,
        openai_api_key=_env("OPENAI_API_KEY") or "",
        model=_env("MODEL") or str(llm.get("model") or ""),
        llm_mode=llm_mode,
        temperature=float(llm.get("temperature", 0)),
        chunk_size=_env_int("CHUNK_SIZE", int(ingest.get("chunk_size", 800))),
        chunk_overlap=_env_int("CHUNK_OVERLAP", int(ingest.get("chunk_overlap", 120))),
        top_k=_env_int("TOP_K", int(retrieval.get("top_k", 5))),
        eval_k=_env_int("EVAL_K", int(evaluation.get("k", 5))),
        min_recall=float(evaluation.get("min_recall", 1.0)),
        eval_dataset=resolve_from_root(
            root, str(evaluation.get("dataset") or "demo/eval/labeled.jsonl")
        ),
        eval_corpus=resolve_from_root(root, str(evaluation.get("corpus") or "demo/corpus")),
        root=root,
    )
    _validate(settings)
    return settings


def resolve_from_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _default_embedding_model(provider: str, model: str) -> str:
    if provider == "hash":
        return model or "hash"
    if provider == "sentence-transformers" and model in {"", "hash"}:
        return "all-MiniLM-L6-v2"
    if provider == "openai" and model in {"", "hash"}:
        return "text-embedding-3-small"
    return model


def _validate(settings: Settings) -> None:
    if not 1 <= settings.port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    if settings.vector_size < 8:
        raise ValueError("EMBEDDING_DIM must be >= 8")
    if settings.chunk_size < 1:
        raise ValueError("CHUNK_SIZE must be positive")
    if settings.chunk_overlap < 0 or settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
    if settings.top_k < 1 or settings.eval_k < 1:
        raise ValueError("TOP_K and eval k must be >= 1")
    if not 0.0 <= settings.min_recall <= 1.0:
        raise ValueError("eval.min_recall must be between 0 and 1")
    if not settings.collection.strip():
        raise ValueError("QDRANT_COLLECTION must not be empty")


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def _section(data: dict, key: str) -> dict:
    value = data.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config section '{key}' must be a mapping")
    return value


def _env(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
