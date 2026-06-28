"""Tests for the ragobserve CLI: version, export (SQLite + Postgres), eval."""
from __future__ import annotations

import json
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from ragobserve.cli import main
from ragobserve.server.db import Store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ev(project: str, stage: str = "retrieval") -> dict:
    return {
        "event_id": uuid.uuid4().hex,
        "trace_id": uuid.uuid4().hex,
        "span_id": uuid.uuid4().hex,
        "project": project,
        "stage": stage,
        "name": stage,
        "start_time": time.time(),
        "end_time": time.time(),
        "duration_ms": 10.0,
        "status": "ok",
        "attributes": {"query": "test?", "response": "answer", "model": "gpt-4o"},
    }


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "cli.db")
    s = Store(path)
    s.ingest_events([_ev("cliproj", "retrieval")])
    # add a generation event on the same trace for eval tests
    t = s.list_traces("cliproj")[0]
    gen = _ev("cliproj", "generation")
    gen["trace_id"] = t["trace_id"]
    s.ingest_events([gen])
    s.close()
    return path


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

def test_version(capsys):
    rc = main(["version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ragobserve" in out
    assert "0." in out


# ---------------------------------------------------------------------------
# export — SQLite
# ---------------------------------------------------------------------------

def test_export_stdout(db, capsys):
    rc = main(["export", "--project", "cliproj", "--backend-store-uri", db])
    assert rc == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert "trace" in data
    assert "events" in data


def test_export_file(db, tmp_path):
    out = str(tmp_path / "out.ndjson")
    rc = main(["export", "--project", "cliproj", "--backend-store-uri", db, "-o", out])
    assert rc == 0
    lines = open(out).read().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["trace"]["project"] == "cliproj"


def test_export_empty_project_returns_0(db, tmp_path):
    out = str(tmp_path / "empty.ndjson")
    rc = main(["export", "--project", "no-such-project", "--backend-store-uri", db, "-o", out])
    assert rc == 0
    assert open(out).read().strip() == ""


def test_export_limit(db, tmp_path):
    out = str(tmp_path / "limited.ndjson")
    rc = main(["export", "--project", "cliproj", "--backend-store-uri", db, "-o", out, "--limit", "1"])
    assert rc == 0
    assert len(open(out).read().splitlines()) == 1


# ---------------------------------------------------------------------------
# export — Postgres
# ---------------------------------------------------------------------------

_requires_pg = pytest.mark.skipif(
    not all(k in __import__("os").environ for k in ("PG_USER", "PG_PASS", "PG_HOST")),
    reason="Postgres env vars not set",
)


@_requires_pg
def test_export_postgres(tmp_path):
    from urllib.parse import quote_plus
    import os

    dsn = (
        f"postgresql://{os.environ['PG_USER']}:{quote_plus(os.environ['PG_PASS'])}"
        f"@{os.environ['PG_HOST']}:{os.environ.get('PG_PORT', '5432')}/{os.environ['DB_NAME']}"
    )
    out = str(tmp_path / "pg_export.ndjson")
    rc = main(["export", "--project", "nonexistent-pg-project", "--backend-store-uri", dsn, "-o", out])
    assert rc == 0  # empty result is fine


# ---------------------------------------------------------------------------
# eval — SQLite (Groq mocked)
# ---------------------------------------------------------------------------

def _fake_groq(*a, **kw) -> MagicMock:
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {
        "choices": [{"message": {"content": '{"score": 0.82, "reason": "looks good"}'}}]
    }
    return m


def test_eval_cli(db, capsys):
    with patch("httpx.post", side_effect=_fake_groq):
        rc = main(["eval", "--project", "cliproj",
                   "--backend-store-uri", db,
                   "--api-key", "fake-groq-key"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "faithfulness=" in out
    assert "0.82" in out


def test_eval_cli_no_traces(db, capsys):
    rc = main(["eval", "--project", "ghost-project",
               "--backend-store-uri", db,
               "--api-key", "fake-groq-key"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "No traces" in out


def test_eval_cli_groq_error(db, capsys):
    """Groq failures are logged but don't crash the command."""
    def boom(*a, **kw):
        raise RuntimeError("Groq down")

    with patch("httpx.post", side_effect=boom):
        rc = main(["eval", "--project", "cliproj",
                   "--backend-store-uri", db,
                   "--api-key", "fake-groq-key"])
    # rc still 0 (partial failure is ok) and error printed
    assert rc == 0


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------

def test_providers(capsys):
    rc = main(["providers"])
    assert rc == 0
    out = capsys.readouterr().out
    assert len(out.strip()) > 0
