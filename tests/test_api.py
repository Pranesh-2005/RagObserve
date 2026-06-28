import os

import pytest
from fastapi.testclient import TestClient

from ragobserve.events import RagEvent, Stage, content_hash
from ragobserve.server.app import create_app

_TEST_KEY = "test-api-key-ragobserve"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGOBSERVE_API_KEY", _TEST_KEY)
    app = create_app(str(tmp_path / "api.db"))
    with TestClient(app, headers={"Authorization": f"Bearer {_TEST_KEY}"}) as c:
        yield c
    app.state.store.close()


@pytest.fixture
def unauth_client(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGOBSERVE_API_KEY", _TEST_KEY)
    app = create_app(str(tmp_path / "unauth.db"))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.state.store.close()


def make_events(trace_id="t1", project="p1"):
    retrieval = RagEvent(
        trace_id=trace_id, project=project, stage=Stage.RETRIEVAL.value, name="retrieval",
        attributes={
            "query": "what is the notice period?",
            "results": [
                {"text": "ninety days notice", "score": 0.91},
                {"text": "unrelated", "score": 0.4},
            ],
            "retriever": "qdrant",
        },
    ).finish()
    generation = RagEvent(
        trace_id=trace_id, project=project, stage=Stage.GENERATION.value, name="gen",
        attributes={"model": "gpt-4o", "response": "90 days", "cost": 0.002},
    ).finish()
    return [retrieval.model_dump(), generation.model_dump()]


def test_ingest_and_query_roundtrip(client):
    r = client.post("/api/events", json={"events": make_events()})
    assert r.status_code == 200 and r.json()["ingested"] == 2

    projects = client.get("/api/projects").json()
    assert projects[0]["name"] == "p1"
    assert projects[0]["traces"] == 1
    assert projects[0]["total_cost"] == pytest.approx(0.002)

    traces = client.get("/api/traces", params={"project": "p1"}).json()
    assert traces[0]["query"] == "what is the notice period?"
    assert traces[0]["model"] == "gpt-4o"
    assert traces[0]["chunk_count"] == 2

    detail = client.get("/api/traces/t1").json()
    assert len(detail["events"]) == 2
    assert detail["trace"]["status"] == "ok"

    assert client.get("/api/traces/nope").status_code == 404


def test_chunk_views_endpoint(client):
    client.post("/api/events", json={"events": make_events()})
    top = client.get("/api/chunks", params={"project": "p1", "view": "top"}).json()
    assert len(top) == 2
    assert top[0]["retrievals"] == 1
    unused = client.get("/api/chunks", params={"project": "p1", "view": "unused"}).json()
    assert unused == []


def test_metrics_endpoint(client):
    client.post("/api/events", json={"events": make_events()})
    relevant = content_hash("ninety days notice")
    r = client.post(
        "/api/ground-truth",
        json={"trace_id": "t1", "project": "p1", "relevant_chunk_ids": [relevant]},
    )
    assert r.status_code == 200
    m = client.get("/api/metrics", params={"project": "p1"}).json()
    assert m["aggregate"]["traces_evaluated"] == 1
    assert m["aggregate"]["mrr"] == 1.0


def test_dashboard_pages_render(client):
    for path in ["/", "/traces", "/chunks", "/metrics", "/traces/t1"]:
        r = client.get(path)
        assert r.status_code == 200
        assert "RAGObserve" in r.text


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

def test_no_key_returns_401(unauth_client):
    assert unauth_client.get("/api/projects").status_code == 401
    assert unauth_client.post("/api/events", json={"events": []}).status_code == 401


def test_wrong_key_returns_401(unauth_client):
    r = unauth_client.get("/api/projects", headers={"Authorization": "Bearer wrong-key"})
    assert r.status_code == 401


def test_x_api_key_header_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGOBSERVE_API_KEY", _TEST_KEY)
    app = create_app(str(tmp_path / "xkey.db"))
    with TestClient(app) as c:
        r = c.get("/api/projects", headers={"X-Api-Key": _TEST_KEY})
        assert r.status_code == 200
    app.state.store.close()


def test_dashboard_pages_no_auth_required(unauth_client):
    # HTML shell pages are unauthenticated — only /api/* routes require the key
    for path in ["/", "/traces", "/chunks", "/metrics"]:
        assert unauth_client.get(path).status_code == 200


def test_api_key_in_app_state(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGOBSERVE_API_KEY", _TEST_KEY)
    app = create_app(str(tmp_path / "state.db"))
    assert app.state.api_key == _TEST_KEY
    app.state.store.close()


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

def test_health_no_auth_required(unauth_client):
    r = unauth_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_returns_version(client):
    r = client.get("/health")
    body = r.json()
    assert "version" in body
    assert body["version"] != ""
