"""SDK client: routes events either directly into a local SQLite store
(no server needed) or to a running RAGObserve tracking server over HTTP,
batched on a background thread with flush-on-exit."""
from __future__ import annotations

import atexit
import queue
import threading
from typing import Any, Dict, List, Optional


class LocalClient:
    """Writes events straight to ./ragobserve.db (or a given path)."""

    def __init__(self, db_path: Optional[str] = None):
        from .server.db import Store
        from .storage import resolve

        self.store = Store(resolve(db_path))

    def log_event(self, event: Dict[str, Any]) -> None:
        self.store.ingest_events([event])

    def log_ground_truth(self, trace_id: str, project: str, relevant_chunk_ids: List[str]) -> None:
        self.store.set_ground_truth(trace_id, project, relevant_chunk_ids)

    def flush(self) -> None:
        pass


class HttpClient:
    """Buffers events and POSTs them to the tracking server in batches."""

    def __init__(self, tracking_uri: str, flush_interval: float = 1.0, batch_size: int = 100):
        import httpx

        self.base = tracking_uri.rstrip("/")
        self._http = httpx.Client(timeout=10.0)
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


def init(project: str = "default", tracking_uri: Optional[str] = None, db_path: Optional[str] = None):
    """Initialize RAGObserve. With no ``tracking_uri`` events are written
    directly to a local SQLite file, MLflow-style. With no ``db_path`` the
    store defaults to a hidden ``./.ragobserve/ragobserve.db``."""
    _config.project = project
    if tracking_uri:
        _config.client = HttpClient(tracking_uri)
    else:
        _config.client = LocalClient(db_path)
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
