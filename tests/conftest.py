"""Hermetic defaults: hash embeddings, extractive answers, in-process Qdrant."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Drop host overrides before any test reads settings. Integration tests still
# see RAGSHIP_INTEGRATION because it is not in this list.
for _name in (
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "MODEL",
    "LLM_PROVIDER",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "HOST",
    "PORT",
    "RAGSHIP_CONFIG",
    "RAGSHIP_ROOT",
    "EVAL_K",
):
    os.environ.pop(_name, None)

os.environ["QDRANT_URL"] = "memory"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["LLM_PROVIDER"] = "extractive"


@pytest.fixture
def client():
    from ragship.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
