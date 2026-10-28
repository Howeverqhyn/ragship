"""Extractive selection and the OpenAI-compatible client, without a live model."""

from __future__ import annotations

import json

import httpx
import pytest

from ragship.config import Settings, load_settings
from ragship.llm.client import (
    ExtractiveAnswerer,
    LLMError,
    OpenAICompatibleClient,
    build_llm,
)


def test_extractive_picks_the_overlapping_sentence():
    answer = ExtractiveAnswerer().answer(
        "Which TCP port does the ragship HTTP API listen on?",
        [
            "The harbor bell rings at dusk. The ragship HTTP API will listen on TCP port 8091.",
        ],
    )
    assert "8091" in answer
    assert "harbor" not in answer


def test_extractive_empty_context():
    assert ExtractiveAnswerer().answer("port", []) == "No matching passages were found."


def test_openai_client_reads_message_content():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "demo-model"
        assert body["temperature"] == 0
        assert request.url.path == "/v1/chat/completions"
        assert "Bearer test-key" == request.headers["authorization"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "  mocked answer  "}}]},
        )

    client = OpenAICompatibleClient(
        base_url="http://llm.test/v1/",
        api_key="test-key",
        model="demo-model",
        transport=httpx.MockTransport(handler),
    )
    assert client.answer("question", ["context passage"]) == "mocked answer"


def test_openai_client_rejects_a_bad_payload():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = OpenAICompatibleClient(
        base_url="http://llm.test/v1",
        api_key="",
        model="demo-model",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMError):
        client.answer("question", ["context"])


def test_ollama_requires_a_model():
    settings = load_settings()
    settings.llm_mode = "openai"
    settings.openai_base_url = "http://ollama:11434/v1"
    settings.model = ""
    with pytest.raises(ValueError, match="MODEL"):
        build_llm(settings)


def test_openai_host_defaults_the_chat_model():
    settings = load_settings()
    settings.llm_mode = "openai"
    settings.openai_base_url = "https://api.openai.com/v1"
    settings.model = ""
    client = build_llm(settings)
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "gpt-4o-mini"


def test_build_llm_extractive_by_default():
    assert isinstance(build_llm(load_settings()), ExtractiveAnswerer)


def test_settings_repr_hides_api_key():
    settings = load_settings()
    settings.openai_api_key = "super-secret-value"
    assert "super-secret-value" not in repr(settings)
    assert isinstance(settings, Settings)
