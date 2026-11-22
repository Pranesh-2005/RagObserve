"""REST API for event ingestion and dashboard queries."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..events import RagEvent, Stage, content_hash, context_items
from . import metrics as M
from . import pricing
from .auth import require_api_key
from .ratelimit import limiter


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
    router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])

    def store(request: Request):
        return request.app.state.store

    @router.post("/events")
    @limiter.limit("500/minute")
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
        scores = {r["metric"]: r["score"] for r in store(request).get_eval_scores(trace_id)}
        t["derived"] = {
            "chunk_utilization": M.chunk_utilization(retrieved, context_ids),
            "reranker_kendall_tau": rerank,
            "triage": M.triage(t, scores),
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

    @router.get("/image")
    def image(request: Request, trace_id: str, path: str):
        """Serve an image the given trace actually retrieved.

        The trace's own logged paths are the whole allowlist. Serving any path the
        caller asks for would make this endpoint an arbitrary local-file read for
        anyone who reaches the dashboard, which on a local-first tool is the
        user's entire home directory — so membership is checked before stat().
        """
        t = store(request).get_trace(trace_id)
        if t is None:
            raise HTTPException(404, "trace not found")
        if path not in _image_paths(t["events"]):
            raise HTTPException(403, "path not referenced by this trace")
        media = IMAGE_TYPES.get(os.path.splitext(path)[1].lower())
        if media is None or not os.path.isfile(path):
            raise HTTPException(404, "image file is gone or not a supported type")
        return FileResponse(path, media_type=media)

    @router.get("/triage")
    def triage(request: Request, project: str, limit: int = Query(200, ge=1, le=2000),
               threshold: float = 0.5):
        """First failing stage per trace, plus counts per verdict."""
        return _triage(store(request), project, limit, threshold)

    @router.get("/compare")
    def compare(request: Request, a: str, b: str, k: int = 10):
        """Side-by-side summary of two projects (e.g. two golden runs)."""
        s = store(request)
        return {"k": k, "a": _summary(s, a, k), "b": _summary(s, b, k)}

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
    @limiter.limit("10/minute")
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

    # --------------------------------------------------- eval (LLM-as-judge)

    @router.post("/eval/{trace_id}")
    @limiter.limit("20/minute")
    def run_eval(
        trace_id: str,
        request: Request,
        provider: str = "groq",
        model: Optional[str] = None,
        backend: str = Query("judge", pattern="^(judge|ragas)$"),
    ):
        """Score faithfulness + answer_relevance on a trace.

        ``backend=judge`` (default) uses the built-in LLM judge on any replay
        provider (key from the server env, e.g. ``GROQ_API_KEY``).
        ``backend=ragas`` uses Ragas metrics (``pip install ragobserve[ragas]``).
        Scores are persisted and returned with the trace via ``GET /api/traces/{id}``.
        """
        from . import llm

        s = store(request)
        t = s.get_trace(trace_id)
        if t is None:
            raise HTTPException(404, "trace not found")
        try:
            if backend == "ragas":
                from ..ragas import evaluate_trace as ragas_eval
                scores = ragas_eval(t)
                model = "ragas"
            else:
                from ..eval import evaluate_trace
                model = model or llm.default_model(provider)
                scores = evaluate_trace(t, model=model, provider=provider)
        except (RuntimeError, ValueError) as e:
            raise HTTPException(502, str(e))

        project = t["trace"].get("project", "default")
        for metric, result in scores.items():
            s.set_eval_score(
                trace_id, project, metric,
                result.get("score"), result.get("reason", ""), model,
            )
        return {"trace_id": trace_id, **scores}

    @router.get("/eval/{trace_id}")
    def get_eval(trace_id: str, request: Request):
        """Return previously computed eval scores for a trace."""
        scores = store(request).get_eval_scores(trace_id)
        return {"trace_id": trace_id, "scores": scores}

    return router


IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}

# Every result list a stage can carry — image hits show up in any of them once a
# reranker or fusion step passes them along.
_RESULT_KEYS = ("results", "chunks", "before", "after")


def _image_paths(events: List[Dict[str, Any]]) -> set:
    """Every image path this trace logged. The allowlist for GET /api/image."""
    paths = set()
    for ev in events:
        attrs = ev.get("attributes") or {}
        lists = [attrs.get(k) for k in _RESULT_KEYS]
        lists.extend((attrs.get("inputs") or {}).values())
        for items in lists:
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                meta = item.get("metadata") or {}
                if meta.get("modality") == "image" and meta.get("path"):
                    paths.add(meta["path"])
    return paths


def _triage(s, project: str, limit: int, threshold: float = 0.5) -> Dict[str, Any]:
    # ponytail: one get_trace per row (N+1 queries); fine at the 2000-row cap
    rows, counts = [], {}
    for t in s.list_traces(project, limit):
        detail = s.get_trace(t["trace_id"])
        if detail is None:
            continue
        v = M.triage(detail, {"faithfulness": t.get("faithfulness_score"),
                              "answer_relevance": t.get("answer_relevance_score")}, threshold)
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
        rows.append({"trace_id": t["trace_id"], "query": t.get("query"), **v})
    return {"counts": counts, "traces": rows}


def _summary(s, project: str, k: int) -> Dict[str, Any]:
    traces = s.list_traces(project, 10_000)
    lat = [t["duration_ms"] for t in traces if t.get("duration_ms") is not None]
    costs = [t["cost"] for t in traces if t.get("cost") is not None]
    n = len(traces)
    return {
        "project": project,
        "traces": n,
        "error_rate": (sum(1 for t in traces if t.get("status") == "error") / n) if n else None,
        "latency_p50_ms": M.percentile(lat, 50),
        "latency_p95_ms": M.percentile(lat, 95),
        "avg_cost": (sum(costs) / len(costs)) if costs else None,
        "total_cost": sum(costs) if costs else None,
        "faithfulness": M._mean([t.get("faithfulness_score") for t in traces]),
        "answer_relevance": M._mean([t.get("answer_relevance_score") for t in traces]),
        "retrieval": M.evaluate_traces(s.traces_with_ground_truth(project), k)["aggregate"],
        "triage": _triage(s, project, 500)["counts"],
    }


MAX_REPLAY_IMAGES = 8
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def _replay_images(events: List[Dict[str, Any]]) -> List[tuple]:
    """Images in what the model saw, as ``(mime, base64)`` for replay.

    Paths are trusted for the same reason ``/api/image`` trusts them: they come
    from this trace's own logged results, filtered to known image types.
    ponytail: capped at 8 images / 5MB each, raise if vision prompts need more.
    """
    import base64

    out = []
    for item in context_items(events):
        meta = item.get("metadata") or {}
        if meta.get("modality") != "image":
            continue
        if meta.get("image_b64"):
            out.append((meta.get("mime") or "image/png", meta["image_b64"]))
        elif meta.get("path"):
            mime = IMAGE_TYPES.get(os.path.splitext(meta["path"])[1].lower())
            if mime and os.path.isfile(meta["path"]) and os.path.getsize(meta["path"]) <= MAX_IMAGE_BYTES:
                with open(meta["path"], "rb") as f:
                    out.append((mime, base64.b64encode(f.read()).decode()))
        if len(out) >= MAX_REPLAY_IMAGES:
            break
    return out


_ids = M.ids_of
