"""Regression tests: async @trace timing, `ui` honouring a passed store,
chunk_retrievals dedupe, judge routed through any provider."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

import ragobserve
from ragobserve.server.db import Store


def test_async_decorator_spans_the_awaited_work(local_client):
    captured = {}

    @ragobserve.trace("retrieval")
    async def retrieve():
        captured["tid"] = ragobserve.current_trace_id()
        await asyncio.sleep(0.05)
        return 7

    assert asyncio.iscoroutinefunction(retrieve)
    assert asyncio.run(retrieve()) == 7
    ev = local_client.store.get_trace(captured["tid"])["events"][0]
    assert ev["duration_ms"] >= 45  # was ~0.03 ms: span closed before the await ran


def test_create_app_uses_passed_store(tmp_path):
    from ragobserve.server.app import create_app

    s = Store(str(tmp_path / "x.db"))
    assert create_app(store=s).state.store is s


def test_cli_open_store_routes_dsn_to_postgres(tmp_path):
    from ragobserve import cli

    with patch("ragobserve.stores.PostgresStore") as pg:
        cli._open_store("postgresql://u:p@h:5432/db")
    pg.assert_called_once_with("postgresql://u:p@h:5432/db")
    assert isinstance(cli._open_store(str(tmp_path / "y.db")), Store)


def _retrieval_event():
    return {
        "event_id": uuid.uuid4().hex, "trace_id": uuid.uuid4().hex, "project": "p",
        "stage": "retrieval", "name": "retrieval", "start_time": time.time(),
        "end_time": time.time(), "status": "ok",
        "attributes": {"query": "q", "results": [{"id": "c1", "text": "a"}, {"id": "c2", "text": "b"}]},
    }


def _cr_count(path):
    with sqlite3.connect(path) as c:
        return c.execute("SELECT COUNT(*) FROM chunk_retrievals").fetchone()[0]


def test_resent_event_does_not_duplicate_chunk_retrievals(tmp_path):
    path = str(tmp_path / "d.db")
    s = Store(path)
    ev = _retrieval_event()
    s.ingest_events([ev])
    s.ingest_events([ev])  # HttpClient retry / MultiStore double write
    s.close()
    assert _cr_count(path) == 2


def test_migration_dedupes_existing_rows(tmp_path):
    path = str(tmp_path / "old.db")
    Store(path).close()
    with sqlite3.connect(path) as c:  # rewind to a pre-migration-2 db holding dupes
        c.execute("DROP INDEX uq_cr_event_chunk")
        c.execute("DELETE FROM schema_versions WHERE version=2")
        for _ in range(3):
            c.execute("INSERT INTO chunk_retrievals(event_id, trace_id, project, content_hash, score, rank)"
                      " VALUES ('e','t','p','c1',0.5,1)")
    Store(path).close()
    assert _cr_count(path) == 1


def _openai_ok(score):
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {"choices": [{"message": {"content": json.dumps({"score": score, "reason": "r"})}}]}
    return m


def test_judge_uses_any_provider():
    from ragobserve.eval import score_answer_relevance

    with patch("httpx.post", return_value=_openai_ok(0.6)) as post:
        r = score_answer_relevance("90 days.", "notice period?", provider="openai", api_key="k")
    assert r["score"] == pytest.approx(0.6)
    url = post.call_args.args[0]
    body = post.call_args.kwargs["json"]
    assert url.startswith("https://api.openai.com/")
    assert body["temperature"] == 0.0 and body["model"] == "gpt-4o-mini"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer k"


def test_judge_default_is_not_retired_groq_model():
    from ragobserve.eval import score_answer_relevance

    with patch("httpx.post", return_value=_openai_ok(1.0)) as post:
        score_answer_relevance("a", "q", api_key="k")
    assert post.call_args.kwargs["json"]["model"] != "llama3-8b-8192"
