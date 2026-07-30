"""Image results: allowlist enforcement on /api/image, and the path collection it uses."""
import base64

import pytest
from fastapi.testclient import TestClient

from ragobserve.server import app as server_app
from ragobserve.server.api import _image_paths
from ragobserve.server.auth import get_api_key


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGOBSERVE_API_KEY", "test-key")
    app = server_app.create_app(str(tmp_path / "t.db"))
    return TestClient(app), app.state.store


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def _png(path):
    path.write_bytes(PNG_1X1)
    return str(path)


def _event(trace_id, img_path, stage="retrieval"):
    return {
        "event_id": f"e-{trace_id}-{stage}",
        "trace_id": trace_id, "span_id": "s1", "project": "mm", "stage": stage,
        "name": stage, "start_time": 1.0, "end_time": 1.1, "duration_ms": 100.0,
        "status": "ok",
        "attributes": {"query": "q", "retriever": "clip", "results": [
            {"id": "img:1", "text": "", "score": 0.4, "source": "a.pdf p.1",
             "metadata": {"modality": "image", "path": img_path}},
        ]},
    }


def test_image_paths_collects_every_result_key():
    attrs = {
        "results": [{"metadata": {"modality": "image", "path": "/a.png"}}],
        "before": [{"metadata": {"modality": "image", "path": "/b.png"}}],
        "after": [{"metadata": {"modality": "image", "path": "/c.png"}}],
        "chunks": [{"metadata": {"modality": "image", "path": "/d.png"}}],
        "inputs": {"clip": [{"metadata": {"modality": "image", "path": "/e.png"}}]},
    }
    assert _image_paths([{"attributes": attrs}]) == {"/a.png", "/b.png", "/c.png", "/d.png", "/e.png"}


def test_image_paths_ignores_text_results():
    attrs = {"results": [{"id": "c1", "text": "hi", "metadata": {"source": "a.pdf"}},
                         {"id": "c2", "text": "yo"}]}
    assert _image_paths([{"attributes": attrs}]) == set()


def test_served_only_when_the_trace_logged_it(client, tmp_path):
    c, store = client
    img = _png(tmp_path / "shown.png")
    store.ingest_events([_event("t1", img)])
    h = {"Authorization": f"Bearer {get_api_key()}"}

    r = c.get("/api/image", params={"trace_id": "t1", "path": img}, headers=h)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG")


def test_untraced_path_is_refused(client, tmp_path):
    """The guard that stops this being an arbitrary local file read."""
    c, store = client
    secret = tmp_path / "secret.png"
    _png(secret)
    store.ingest_events([_event("t1", _png(tmp_path / "shown.png"))])
    h = {"Authorization": f"Bearer {get_api_key()}"}

    r = c.get("/api/image", params={"trace_id": "t1", "path": str(secret)}, headers=h)
    assert r.status_code == 403
    # ..and traversal gets the same treatment: it is simply not in the allowlist
    r = c.get("/api/image",
              params={"trace_id": "t1", "path": str(tmp_path / ".." / "etc" / "passwd")},
              headers=h)
    assert r.status_code == 403


def test_image_requires_auth(client, tmp_path):
    c, store = client
    img = _png(tmp_path / "shown.png")
    store.ingest_events([_event("t1", img)])
    assert c.get("/api/image", params={"trace_id": "t1", "path": img}).status_code == 401


def test_missing_file_is_404_not_500(client, tmp_path):
    c, store = client
    gone = str(tmp_path / "deleted.png")
    store.ingest_events([_event("t1", gone)])
    h = {"Authorization": f"Bearer {get_api_key()}"}
    assert c.get("/api/image", params={"trace_id": "t1", "path": gone}, headers=h).status_code == 404


def test_non_image_extension_refused_even_if_traced(client, tmp_path):
    """A logged path is not a licence to serve any file type."""
    c, store = client
    env = tmp_path / "logged.env"
    env.write_text("SECRET=1")
    store.ingest_events([_event("t1", str(env))])
    h = {"Authorization": f"Bearer {get_api_key()}"}
    assert c.get("/api/image", params={"trace_id": "t1", "path": str(env)}, headers=h).status_code == 404


def test_unknown_trace_is_404(client):
    c, _ = client
    h = {"Authorization": f"Bearer {get_api_key()}"}
    r = c.get("/api/image", params={"trace_id": "nope", "path": "/a.png"}, headers=h)
    assert r.status_code == 404
