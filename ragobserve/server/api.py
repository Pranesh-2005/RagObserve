"""REST API for event ingestion and dashboard queries."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from ..events import RagEvent, Stage, content_hash
from . import metrics as M
from . import pricing


class EventBatch(BaseModel):
    events: List[Dict[str, Any]]


class GroundTruth(BaseModel):
    trace_id: str
    project: str = "default"
    relevant_chunk_ids: List[str]


class GenerateRequest(BaseModel):
    provider: str = "anthropic"
    model: Optional[str] = None
    # Either replay a trace's captured context, or send an ad-hoc prompt.
    trace_id: Optional[str] = None
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    project: str = "default"
    max_tokens: int = 1024


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    def store(request: Request):
        return request.app.state.store

    @router.post("/events")
    def ingest(batch: EventBatch, request: Request):
        n = store(request).ingest_events(batch.events)
        return {"ingested": n}

    @router.post("/ground-truth")
    def ground_truth(gt: GroundTruth, request: Request):
        store(request).set_ground_truth(gt.trace_id, gt.project, gt.relevant_chunk_ids)
        return {"ok": True}

    @router.get("/projects")
    def projects(request: Request):
        return store(request).list_projects()

    @router.get("/traces")
    def traces(request: Request, project: Optional[str] = None, limit: int = 200):
        return store(request).list_traces(project, limit)

    @router.get("/traces/{trace_id}")
    def trace_detail(trace_id: str, request: Request):
        t = store(request).get_trace(trace_id)
        if t is None:
            raise HTTPException(404, "trace not found")
        # enrich with derived per-trace metrics
        retrieved, context_ids, rerank = [], [], None
        for ev in t["events"]:
            a = ev["attributes"]
            if ev["stage"] == "retrieval":
                retrieved = _ids(a.get("results"))
            elif ev["stage"] == "reranking":
                rerank = M.kendall_tau(_ids(a.get("before")), _ids(a.get("after")))
            elif ev["stage"] == "context_assembly":
                context_ids = _ids(a.get("chunks"))
        t["derived"] = {
            "chunk_utilization": M.chunk_utilization(retrieved, context_ids),
            "reranker_kendall_tau": rerank,
        }
        return t

    @router.get("/chunks")
    def chunks(
        request: Request,
        project: str,
        view: str = Query("top", pattern="^(top|unused|duplicates)$"),
        limit: int = 100,
    ):
        return store(request).chunk_views(project, view, limit)

    @router.get("/metrics")
    def metrics(request: Request, project: str, k: int = 10):
        pairs = store(request).traces_with_ground_truth(project)
        return M.evaluate_traces(pairs, k)

    # --------------------------------------------------- cost & generation views

    @router.get("/costs")
    def costs(request: Request, project: str):
        return store(request).cost_summary(project)

    @router.get("/generations")
    def generations(request: Request, project: str, limit: int = 200):
        return store(request).list_generations(project, limit)

    @router.get("/providers")
    def providers():
        from . import llm
        return llm.available_providers()

    @router.post("/generate")
    def generate(req: GenerateRequest, request: Request):
        """Live "replay generation": run an LLM over a trace's captured context
        (or an ad-hoc prompt), log it back into the trace, and return the answer
        with computed cost — so the new generation shows up in the dashboards."""
        from . import llm

        s = store(request)
        prompt = req.prompt
        system = req.system_prompt
        trace_id = req.trace_id
        project = req.project

        if trace_id and not prompt:
            ctx = s.get_generation_context(trace_id)
            if ctx is None:
                raise HTTPException(404, "trace not found")
            prompt = ctx.get("final_prompt") or ctx.get("query")
            system = system or ctx.get("system_prompt")
            project = s.get_trace(trace_id)["trace"].get("project", project)
        if not prompt:
            raise HTTPException(400, "nothing to generate from: pass prompt or a trace_id with context")

        import time as _time
        t0 = _time.time()
        try:
            result = llm.generate(req.provider, req.model or "", prompt, system, req.max_tokens)
        except llm.ProviderError as e:
            raise HTTPException(502, str(e))
        duration_ms = (_time.time() - t0) * 1000.0

        cost = pricing.estimate_cost(result["model"], result.get("input_tokens"),
                                     result.get("output_tokens"))

        # Log the replay as a generation event attached to the same trace.
        ev = RagEvent(
            project=project,
            trace_id=trace_id or RagEvent().trace_id,
            stage=Stage.GENERATION.value,
            name="generation (replay)",
            start_time=t0,
            end_time=t0 + duration_ms / 1000.0,
            duration_ms=duration_ms,
            attributes={
                "model": result["model"], "provider": result["provider"],
                "prompt": prompt, "system_prompt": system, "response": result["text"],
                "input_tokens": result.get("input_tokens"),
                "output_tokens": result.get("output_tokens"),
                "cost": cost, "replayed": True,
            },
        )
        s.ingest_events([ev.model_dump()])

        return {
            "trace_id": ev.trace_id,
            "model": result["model"], "provider": result["provider"],
            "response": result["text"],
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cost": cost, "duration_ms": duration_ms,
        }

    return router


def _ids(results: Optional[List[Any]]) -> List[str]:
    out = []
    for item in results or []:
        if isinstance(item, dict):
            cid = item.get("id") or (content_hash(item["text"]) if item.get("text") else None)
            if cid:
                out.append(cid)
    return out
