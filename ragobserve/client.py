"""SDK client: routes events either directly into a local SQLite store
(no server needed) or to a running RAGObserve tracking server over HTTP,
batched on a background thread with flush-on-exit."""
from __future__ import annotations

import atexit
import queue
import threading
from typing import Any, Dict, List, Optional


class LocalClient:
    """Writes events straight to ./ragobserve.db (or a given path).

    In a running asyncio loop log_event offloads the SQLite write to the
    default thread-pool executor so it never blocks the event loop. In
    synchronous code (tests, scripts) it writes inline."""

    def __init__(self, db_path: Optional[str] = None, store: Optional[object] = None):
        if store is not None:
            self.store = store
        else:
            from .server.db import Store
            from .storage import resolve

            self.store = Store(resolve(db_path))

    def log_event(self, event: Dict[str, Any]) -> None:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.store.ingest_events([event])
            return
        loop.run_in_executor(None, self.store.ingest_events, [event])

    def log_ground_truth(self, trace_id: str, project: str, relevant_chunk_ids: List[str]) -> None:
        self.store.set_ground_truth(trace_id, project, relevant_chunk_ids)

    def flush(self) -> None:
        pass


class HttpClient:
    """Buffers events and POSTs them to the tracking server in batches."""

    def __init__(self, tracking_uri: str, api_key: str = "",
                 flush_interval: float = 1.0, batch_size: int = 100):
        import httpx

        self.base = tracking_uri.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(timeout=10.0, headers=headers)
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._batch_size = batch_size
        self._interval = flush_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="ragobserve-flush")
        self._thread.start()
        atexit.register(self.flush)

    def log_event(self, event: Dict[str, Any]) -> None:
        self._queue.put(event)

    def log_ground_truth(self, trace_id: str, project: str, relevant_chunk_ids: List[str]) -> None:
        self._http.post(
            f"{self.base}/api/ground-truth",
            json={"trace_id": trace_id, "project": project, "relevant_chunk_ids": relevant_chunk_ids},
        )

    def _drain(self) -> List[Dict[str, Any]]:
        batch = []
        while len(batch) < self._batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _send(self, batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
        try:
            self._http.post(f"{self.base}/api/events", json={"events": batch})
        except Exception:
            # observability must never crash the instrumented app
            pass

    def _worker(self) -> None:
        while not self._stop.wait(self._interval):
            self._send(self._drain())

    def flush(self) -> None:
        while True:
            batch = self._drain()
            if not batch:
                break
            self._send(batch)


class _Config:
    project: str = "default"
    client: Optional[object] = None


_config = _Config()


def init(
    project: str = "default",
    tracking_uri: Optional[str] = None,
    db_path: Optional[str] = None,
    store: Optional[object] = None,
    api_key: str = "",
):
    """Initialize RAGObserve.

    Priority: ``tracking_uri`` > ``store`` > ``db_path`` > local default.

    * ``tracking_uri`` — send events to a remote RAGObserve server over HTTP.
    * ``api_key``      — Bearer token for the remote server (required when auth is enabled).
    * ``store``        — use any ``BaseStore``-compatible backend directly
                         (PostgresStore, FileStore, MultiStore, or custom).
    * ``db_path``      — local SQLite file path (default: hidden .ragobserve/ragobserve.db).
    """
    _config.project = project
    if tracking_uri:
        _config.client = HttpClient(tracking_uri, api_key=api_key)
    else:
        _config.client = LocalClient(db_path=db_path, store=store)
    return _config.client


def get_client():
    if _config.client is None:
        init()
    return _config.client


def get_project() -> str:
    return _config.project


def flush() -> None:
    if _config.client is not None:
        _config.client.flush()


def serve(host: str = "127.0.0.1", port: int = 5601, db_path: Optional[str] = None,
          api_key: Optional[str] = None) -> None:
    """Start the RAGObserve dashboard. Equivalent to ``ragobserve ui``."""
    import os
    import uvicorn

    from .server.app import create_app
    from .server.auth import get_api_key

    if api_key:
        os.environ["RAGOBSERVE_API_KEY"] = api_key

    key = get_api_key()
    app = create_app(db_path)
    print(f"RAGObserve UI  → http://{host}:{port}/?key={key}")
    print(f"API key        → {key}")
    print("Set RAGOBSERVE_API_KEY env var to use a fixed key.")
    uvicorn.run(app, host=host, port=port, log_level="warning")
