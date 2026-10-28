"""Extractive answers, and one OpenAI-compatible chat client.

The client speaks POST {OPENAI_BASE_URL}/chat/completions. Ollama and the
OpenAI API are the same call with a different base URL, key, and model.
When no base URL is configured, answers are sentences copied from the chunks.
"""

from __future__ import annotations

import httpx

from ragship import __version__
from ragship.config import Settings
from ragship.text import split_sentences, tokenize


class LLMError(RuntimeError):
    """The chat completion call failed."""


class ExtractiveAnswerer:
    mode = "extractive"

    def answer(self, query: str, contexts: list[str]) -> str:
        if not contexts:
            return "No matching passages were found."
        query_tokens = set(tokenize(query))
        best = ""
        best_score = 0.0
        for context in contexts:
            for sentence in split_sentences(context):
                score = _overlap(query_tokens, sentence)
                if score > best_score:
                    best_score = score
                    best = sentence
        if best_score <= 0:
            return contexts[0].strip()
        return best


class OpenAICompatibleClient:
    """One chat-completions client for Ollama and hosted OpenAI-compatible APIs."""

    mode = "openai"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0,
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_BASE_URL")
        if not model:
            raise ValueError("MODEL is required for this OPENAI_BASE_URL")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.transport = transport

    def answer(self, query: str, contexts: list[str]) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the context. "
                        "If the context does not contain the answer, say you do not know."
                    ),
                },
                {"role": "user", "content": _prompt(query, contexts)},
            ],
        }
        body = _post_json(
            url=f"{self.base_url}/chat/completions",
            payload=payload,
            api_key=self.api_key,
            timeout=self.timeout,
            transport=self.transport,
        )
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("language model returned an unexpected response") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("language model returned an empty answer")
        return content.strip()


def build_llm(settings: Settings) -> ExtractiveAnswerer | OpenAICompatibleClient:
    if settings.llm_mode == "extractive":
        return ExtractiveAnswerer()
    if settings.llm_mode == "openai":
        return OpenAICompatibleClient(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=_chat_model(settings),
            temperature=settings.temperature,
        )
    raise ValueError(f"unknown llm mode: {settings.llm_mode}")


def _chat_model(settings: Settings) -> str:
    if settings.model:
        return settings.model
    if "api.openai.com" in settings.openai_base_url.lower():
        return "gpt-4o-mini"
    raise ValueError("MODEL is required when OPENAI_BASE_URL is not the OpenAI API")


def _prompt(query: str, contexts: list[str]) -> str:
    if contexts:
        blocks = [f"[{index}] {context}" for index, context in enumerate(contexts, start=1)]
        joined = "\n\n".join(blocks)
    else:
        joined = "(no context)"
    return f"Context:\n{joined}\n\nQuestion: {query}"


def _overlap(query_tokens: set[str], sentence: str) -> float:
    if not query_tokens:
        return 0.0
    hits = sum(1 for token in tokenize(sentence) if token in query_tokens)
    return hits / (len(query_tokens) ** 0.5)


def _post_json(
    *,
    url: str,
    payload: dict,
    api_key: str,
    timeout: float,
    transport: httpx.BaseTransport | None,
) -> dict:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"ragship/{__version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise LLMError(f"language model request failed: {exc}") from exc
    if response.status_code >= 400:
        detail = response.text.strip().replace("\n", " ")[:300]
        raise LLMError(f"language model request failed with HTTP {response.status_code}: {detail}")
    try:
        body = response.json()
    except ValueError as exc:
        raise LLMError("language model response was not JSON") from exc
    if not isinstance(body, dict):
        raise LLMError("language model response was not a JSON object")
    return body
