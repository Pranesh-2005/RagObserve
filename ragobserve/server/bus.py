"""Thread-safe event bus: bridges sync Store.ingest_events → async WebSocket clients.

Call ``bus.set_loop(loop)`` once at server startup.
Call ``bus.publish(event)`` from any thread after writing an event.
Each WebSocket connection subscribes with ``bus.subscribe()`` / ``bus.unsubscribe()``.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Set

_loop: asyncio.AbstractEventLoop | None = None
_subscribers: Set[asyncio.Queue] = set()


def set_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


def subscribe() -> "asyncio.Queue[Dict[str, Any]]":
    q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
    _subscribers.add(q)
    return q


def unsubscribe(q: "asyncio.Queue[Dict[str, Any]]") -> None:
    _subscribers.discard(q)


def publish(event: Dict[str, Any]) -> None:
    """Safe to call from any thread. No-op when no subscribers or loop not set."""
    if _loop is not None and _loop.is_running() and _subscribers:
        for q in list(_subscribers):
            _loop.call_soon_threadsafe(q.put_nowait, event)
