"""HTTP contract for /health, /ingest, and /query."""

from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient
from tests.pdfutil import make_pdf

from ragship.llm.client import OpenAICompatibleClient


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["qdrant"] == "memory"
    assert body["embedding_provider"] == "hash"
    assert body["llm_mode"] == "extractive"
    assert body["version"]


def test_ingest_json_and_extractive_query(client):
    ingested = client.post(
        "/ingest",
        json={
            "source": "facts.txt",
            "text": (
                "The harbor bell rings at dusk. "
                "The ragship HTTP API will listen on TCP port 8091. "
                "Seagulls follow the ferry."
            ),
        },
    )
    assert ingested.status_code == 200
    payload = ingested.json()
    assert payload["source"] == "facts.txt"
    assert payload["chunks"] == 1
    assert payload["ids"] == ["facts.txt#0"]

    response = client.post(
        "/query",
        json={"query": "Which TCP port does the ragship HTTP API listen on?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "extractive"
    assert "8091" in body["answer"]
    assert body["sources"][0]["source"] == "facts.txt"
    assert body["sources"][0]["chunk_id"] == "facts.txt#0"


def test_ingest_markdown_txt_and_pdf(client):
    markdown = client.post(
        "/ingest",
        files={"file": ("notes.md", b"# Notes\n\nMarkdown kelp forests sway.\n", "text/markdown")},
    )
    plain = client.post(
        "/ingest",
        files={"file": ("notes.txt", b"Plain kelp forests are underwater.\n", "text/plain")},
    )
    pdf = client.post(
        "/ingest",
        files={
            "file": (
                "notes.pdf",
                make_pdf("RagshipPdfToken ZEBRAUNIQUE sails."),
                "application/pdf",
            )
        },
    )
    assert markdown.status_code == 200
    assert plain.status_code == 200
    assert pdf.status_code == 200
    found = client.post("/query", json={"query": "Where is ZEBRAUNIQUE?"})
    assert found.status_code == 200
    assert found.json()["sources"][0]["source"] == "notes.pdf"
    assert "ZEBRAUNIQUE" in found.json()["sources"][0]["text"]


def test_form_text_field(client):
    response = client.post(
        "/ingest",
        data={"text": "Form kelp forests drift with the tide.", "source": "kelp.txt"},
    )
    assert response.status_code == 200
    assert response.json()["source"] == "kelp.txt"


def test_source_name_strips_directories(client):
    response = client.post(
        "/ingest",
        json={"text": "A basename only note about orchids.", "source": "../../etc/orchids.txt"},
    )
    assert response.status_code == 200
    assert response.json()["source"] == "orchids.txt"


def test_missing_suffix_becomes_txt(client):
    response = client.post(
        "/ingest",
        json={"text": "No suffix here about orchids.", "source": "note"},
    )
    assert response.status_code == 200
    assert response.json()["source"] == "note.txt"


def test_unsupported_type_and_empty_document(client):
    rejected = client.post(
        "/ingest",
        files={"file": ("sheet.docx", b"not a real document", "application/octet-stream")},
    )
    empty = client.post("/ingest", files={"file": ("empty.txt", b"   \n", "text/plain")})
    punctuation = client.post("/ingest", json={"text": "???", "source": "marks.txt"})
    assert rejected.status_code == 415
    assert empty.status_code == 400
    assert punctuation.status_code == 400


def test_whitespace_query_is_rejected(client):
    response = client.post("/query", json={"query": "   "})
    assert response.status_code == 422


def test_query_with_empty_store(client):
    response = client.post("/query", json={"query": "anything indexed"})
    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert body["answer"] == "No matching passages were found."
    assert body["mode"] == "extractive"


def test_reingest_replaces_previous_chunks(client):
    first = client.post(
        "/ingest",
        json={"text": "alpha bravo uniquealpha token", "source": "swap.txt"},
    )
    assert first.status_code == 200
    second = client.post(
        "/ingest",
        json={"text": "charlie delta uniquecharlie token", "source": "swap.txt"},
    )
    assert second.status_code == 200
    stale = client.post("/query", json={"query": "uniquealpha"})
    fresh = client.post("/query", json={"query": "uniquecharlie"})
    for hit in stale.json()["sources"]:
        if hit["source"] == "swap.txt":
            assert "uniquealpha" not in hit["text"]
    assert fresh.json()["sources"][0]["text"] == "charlie delta uniquecharlie token"


def test_upload_limit(client, monkeypatch):
    monkeypatch.setattr("ragship.api.routes.MAX_UPLOAD_BYTES", 8)
    response = client.post(
        "/ingest",
        files={"file": ("a.txt", b"0123456789", "text/plain")},
    )
    assert response.status_code == 413


def test_query_uses_openai_compatible_client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MODEL", "demo-model")

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "model says hi"}}]})

    from ragship.api.app import create_app

    app = create_app()
    app.state.llm = OpenAICompatibleClient(
        base_url="http://llm.test/v1",
        api_key="test-key",
        model="demo-model",
        transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as test_client:
        ingested = test_client.post(
            "/ingest",
            json={"text": "Ragship listens on port 8091.", "source": "a.txt"},
        )
        assert ingested.status_code == 200
        response = test_client.post("/query", json={"query": "Which port?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "model says hi"
    assert body["mode"] == "openai"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "demo-model"
    assert "8091" in seen["body"]["messages"][1]["content"]


def test_llm_http_error_is_bad_gateway(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("MODEL", "demo-model")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream blew up")

    from ragship.api.app import create_app

    app = create_app()
    app.state.llm = OpenAICompatibleClient(
        base_url="http://llm.test/v1",
        api_key="",
        model="demo-model",
        transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as test_client:
        test_client.post("/ingest", json={"text": "Some indexed fact.", "source": "a.txt"})
        response = test_client.post("/query", json={"query": "fact"})
    assert response.status_code == 502
    assert "500" in response.json()["detail"]
