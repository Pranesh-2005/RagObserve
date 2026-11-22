"""FastAPI app factory: serves both the REST API and the dashboard."""
from __future__ import annotations

import asyncio
import json
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .api import build_router
from .auth import get_api_key, require_api_key
from . import bus
from .db import Store
from .ratelimit import limiter

_HERE = os.path.dirname(os.path.abspath(__file__))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # ponytail: in-process bus only works with a single uvicorn worker.
    # Multi-worker deployments need a Redis pub/sub bus — add when needed.
    bus.set_loop(asyncio.get_event_loop())
    yield


def create_app(db_path: str | None = None, store=None) -> FastAPI:
    """``store`` takes any backend with the dashboard query methods (Store,
    PostgresStore); otherwise a SQLite Store is opened at ``db_path``."""
    from ..storage import resolve

    app = FastAPI(title="RAGObserve", docs_url="/api/docs", lifespan=_lifespan)
    app.state.store = store if store is not None else Store(resolve(db_path))
    app.state.api_key = get_api_key()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.include_router(build_router())
    app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")
    templates = Jinja2Templates(directory=os.path.join(_HERE, "templates"))

    def page(request: Request, name: str, **ctx):
        ctx["page"] = name
        return templates.TemplateResponse(request, f"{name}.html", ctx)

    @app.get("/health")
    def health():
        try:
            from importlib.metadata import version
            v = version("ragobserve")
        except Exception:
            v = "unknown"
        return JSONResponse({"status": "ok", "version": v})

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        return page(request, "dashboard")

    @app.get("/traces", response_class=HTMLResponse)
    def traces(request: Request, project: str = ""):
        return page(request, "traces", project=project)

    @app.get("/traces/{trace_id}", response_class=HTMLResponse)
    def trace_detail(request: Request, trace_id: str):
        return page(request, "trace_detail", trace_id=trace_id)

    @app.get("/chunks", response_class=HTMLResponse)
    def chunks(request: Request, project: str = ""):
        return page(request, "chunks", project=project)

    @app.get("/metrics", response_class=HTMLResponse)
    def metrics(request: Request, project: str = ""):
        return page(request, "metrics", project=project)

    @app.get("/generations", response_class=HTMLResponse)
    def generations(request: Request, project: str = ""):
        return page(request, "generations", project=project)

    @app.post("/v1/traces", dependencies=[Depends(require_api_key)])
    @limiter.limit("500/minute")
    async def otlp_traces(request: Request):
        """OTLP/HTTP trace receiver (JSON, or protobuf with opentelemetry-proto)."""
        from . import otel

        ctype = request.headers.get("content-type", "")
        body = await request.body()
        if "protobuf" in ctype:
            try:
                payload = otel.decode_protobuf(body)
            except ImportError:
                return JSONResponse(
                    {"error": "protobuf OTLP needs `pip install ragobserve[otel]` on the server, "
                              "or set OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/json"}, 415)
        else:
            try:
                payload = json.loads(body or b"{}")
            except ValueError:
                return JSONResponse({"error": "body is not OTLP JSON"}, 400)
        events = otel.otlp_to_events(payload, request.headers.get("x-ragobserve-project"))
        if events:
            await run_in_threadpool(app.state.store.ingest_events, events)
        return JSONResponse({"partialSuccess": {}})

    @app.websocket("/ws/traces")
    async def ws_traces(websocket: WebSocket, key: str = "", project: str = ""):
        """Live trace feed. Connect with ?key=<api_key>&project=<project>.

        Receives JSON messages::

            {"type": "event",  "data": {...}}   # new event ingested
            {"type": "ping"}                     # keepalive every ~30 s
        """
        await websocket.accept()
        if not key or not secrets.compare_digest(key, get_api_key()):
            await websocket.close(code=4401, reason="Unauthorized")
            return
        q = bus.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping"})
                    continue
                if not project or event.get("project") == project:
                    await websocket.send_json({"type": "event", "data": event})
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            bus.unsubscribe(q)

    return app
