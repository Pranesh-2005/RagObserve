"""WebSocket /ws/traces tests — auth, connection, event delivery."""
from __future__ import annotations

import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ragobserve.server.app import create_app
from ragobserve.server import bus as _bus

_KEY = "ws-test-key-ragobserve"


@pytest.fixture(autouse=True)
def reset_bus():
    """Clear bus subscribers between tests to avoid cross-test contamination."""
    _bus._subscribers.clear()
    yield
    _bus._subscribers.clear()


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGOBSERVE_API_KEY", _KEY)
    return create_app(str(tmp_path / "ws.db"))


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c
    app.state.store.close()


def _ev(project: str = "wsproj") -> dict:
    return {
        "event_id": uuid.uuid4().hex,
        "trace_id": uuid.uuid4().hex,
        "span_id": uuid.uuid4().hex,
        "project": project,
        "stage": "retrieval",
        "name": "retrieval",
        "start_time": time.time(),
        "end_time": time.time(),
        "duration_ms": 5.0,
        "status": "ok",
        "attributes": {"query": "hello"},
    }


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_ws_rejects_wrong_key(client):
    with pytest.raises((WebSocketDisconnect, Exception)):
        with client.websocket_connect("/ws/traces?key=wrong-key") as ws:
            ws.receive_json()


def test_ws_rejects_missing_key(client):
    with pytest.raises((WebSocketDisconnect, Exception)):
        with client.websocket_connect("/ws/traces") as ws:
            ws.receive_json()


def test_ws_accepts_correct_key(client):
    # Should connect and close cleanly — no exception
    with client.websocket_connect(f"/ws/traces?key={_KEY}") as ws:
        ws.close()


# ---------------------------------------------------------------------------
# Event delivery
# ---------------------------------------------------------------------------

def test_ws_receives_event_after_ingest(client):
    """Background thread posts an event; main thread reads it from WebSocket."""
    ev = _ev(project="live")

    def post():
        time.sleep(0.15)
        client.post(
            "/api/events",
            json={"events": [ev]},
            headers={"Authorization": f"Bearer {_KEY}"},
        )

    threading.Thread(target=post, daemon=True).start()

    with client.websocket_connect(f"/ws/traces?key={_KEY}&project=live") as ws:
        msg = ws.receive_json()

    assert msg["type"] == "event"
    assert msg["data"]["trace_id"] == ev["trace_id"]
    assert msg["data"]["project"] == "live"


def test_ws_filters_by_project(client):
    """Events from a different project are NOT forwarded to a project-filtered socket."""
    ev_other = _ev(project="other-proj")
    ev_mine = _ev(project="my-proj")

    # Post "other" event first, then "mine" shortly after.
    # The WS is subscribed to "my-proj" — it should only get the second event.
    def post():
        time.sleep(0.1)
        client.post(
            "/api/events",
            json={"events": [ev_other]},
            headers={"Authorization": f"Bearer {_KEY}"},
        )
        time.sleep(0.1)
        client.post(
            "/api/events",
            json={"events": [ev_mine]},
            headers={"Authorization": f"Bearer {_KEY}"},
        )

    threading.Thread(target=post, daemon=True).start()

    with client.websocket_connect(f"/ws/traces?key={_KEY}&project=my-proj") as ws:
        msg = ws.receive_json()

    assert msg["data"]["project"] == "my-proj"
    assert msg["data"]["trace_id"] == ev_mine["trace_id"]


def test_ws_no_project_filter_receives_all(client):
    """An unfiltered WS (no project param) receives events from any project."""
    ev = _ev(project="anything")

    def post():
        time.sleep(0.1)
        client.post(
            "/api/events",
            json={"events": [ev]},
            headers={"Authorization": f"Bearer {_KEY}"},
        )

    threading.Thread(target=post, daemon=True).start()

    with client.websocket_connect(f"/ws/traces?key={_KEY}") as ws:
        msg = ws.receive_json()

    assert msg["type"] == "event"
    assert msg["data"]["trace_id"] == ev["trace_id"]
