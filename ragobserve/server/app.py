"""FastAPI app factory: serves both the REST API and the dashboard."""
from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .api import build_router
from .db import Store

_HERE = os.path.dirname(os.path.abspath(__file__))


def create_app(db_path: str | None = None) -> FastAPI:
    from ..storage import resolve

    app = FastAPI(title="RAGObserve", docs_url="/api/docs")
    app.state.store = Store(resolve(db_path))

    app.include_router(build_router())
    app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")
    templates = Jinja2Templates(directory=os.path.join(_HERE, "templates"))

    def page(request: Request, name: str, **ctx):
        ctx["page"] = name
        return templates.TemplateResponse(request, f"{name}.html", ctx)

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

    return app
