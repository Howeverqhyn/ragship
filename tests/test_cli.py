"""`python -m ragship` serve and eval entry points."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    env["QDRANT_URL"] = "http://127.0.0.1:9"
    env["EMBEDDING_PROVIDER"] = "hash"
    env["LLM_PROVIDER"] = "extractive"
    env.pop("OPENAI_BASE_URL", None)
    env.pop("OPENAI_API_KEY", None)
    return env


def test_eval_cli_passes_with_unreachable_qdrant():
    result = subprocess.run(
        [sys.executable, "-m", "ragship", "eval"],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "recall@5: 1.000" in result.stdout


def test_eval_cli_fails_on_a_miss(tmp_path: Path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(
        '{"query": "zzzz uniquequerytoken", "relevant_ids": ["missing.md"]}\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "ragship", "eval", "--dataset", str(dataset), "--k", "5"],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "recall@5: 0.000" in result.stdout


def test_cli_without_a_command_exits_with_usage():
    result = subprocess.run(
        [sys.executable, "-m", "ragship"],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "serve" in result.stderr
    assert "eval" in result.stderr


def test_serve_command_invokes_uvicorn(monkeypatch):
    captured = {}

    def fake_run(app, host, port, **_kwargs):
        captured["title"] = app.title
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_run)
    from ragship.__main__ import main

    assert main(["serve", "--host", "127.0.0.1", "--port", "8091"]) == 0
    assert captured == {"title": "ragship", "host": "127.0.0.1", "port": 8091}
