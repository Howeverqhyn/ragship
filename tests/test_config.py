"""Environment variables override configs/default.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ragship.config import load_settings


def test_yaml_defaults(monkeypatch):
    for name in ("QDRANT_URL", "EMBEDDING_PROVIDER", "LLM_PROVIDER", "OPENAI_BASE_URL", "MODEL"):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings()
    assert settings.qdrant_url == "memory"
    assert settings.embedding_provider == "hash"
    assert settings.llm_mode == "extractive"
    assert settings.port == 8091
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.top_k == 5
    assert settings.openai_api_key == ""
    assert settings.openai_base_url == ""


def test_env_selects_openai_mode(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MODEL", "gpt-4o-mini")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    settings = load_settings()
    assert settings.llm_mode == "openai"
    assert settings.model == "gpt-4o-mini"
    assert settings.openai_api_key == "test-key"


def test_explicit_extractive_wins_over_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://ollama:11434/v1")
    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    settings = load_settings()
    assert settings.llm_mode == "extractive"


def test_sentence_transformer_model_default(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    settings = load_settings()
    assert settings.embedding_provider == "sentence-transformers"
    assert settings.embedding_model == "all-MiniLM-L6-v2"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mystery")
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        load_settings()


def test_overlap_larger_than_chunk_is_rejected(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "10")
    monkeypatch.setenv("CHUNK_OVERLAP", "10")
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        load_settings()


def test_compose_and_ci_files():
    root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((root / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert "qdrant" in services
    assert "api" in services
    assert services["ollama"]["profiles"] == ["ollama"]
    assert "8091:8091" in services["api"]["ports"]
    assert services["api"]["environment"]["QDRANT_URL"] == "http://qdrant:6333"
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "ruff check" in workflow
    assert "pytest" in workflow
    example = (root / ".env.example").read_text(encoding="utf-8")
    for name in ("QDRANT_URL", "EMBEDDING_PROVIDER", "OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL"):
        assert name in example
