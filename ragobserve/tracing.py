"""Tracing primitives: ``trace`` works as a context manager and a decorator,
with nesting via contextvars producing parent/child spans, plus convenience
loggers that emit correctly-shaped stage events so the dashboard views light
up without users learning the event schema."""
from __future__ import annotations

import contextvars
import functools
import time
from typing import Any, Dict, List, Optional

from . import client as _client
from .events import RagEvent, Stage, STAGES, new_id, normalize_result

_current_span: "contextvars.ContextVar[Optional[RagEvent]]" = contextvars.ContextVar(
    "ragobserve_current_span", default=None
)


def current_trace_id() -> Optional[str]:
    span = _current_span.get()
    return span.trace_id if span else None


def _start_span(name: str, stage: str, attributes: Dict[str, Any]) -> RagEvent:
    parent = _current_span.get()
    ev = RagEvent(
        project=_client.get_project(),
        name=name,
        stage=stage,
        attributes=attributes,
    )
    if parent is not None:
        ev.trace_id = parent.trace_id
        ev.parent_span_id = ev.parent_span_id or parent.span_id
    return ev


class _TraceHandle:
    """Returned by ``trace(...)`` — usable as ``with`` block or ``@`` decorator."""

    def __init__(self, name: str, stage: str, attributes: Dict[str, Any]):
        self._name = name
        self._stage = stage
        self._attributes = attributes
        self._span: Optional[RagEvent] = None
        self._token = None

    # -- context manager -------------------------------------------------
    def __enter__(self) -> RagEvent:
        self._span = _start_span(self._name, self._stage, dict(self._attributes))
        self._token = _current_span.set(self._span)
        return self._span

    def __exit__(self, exc_type, exc, tb) -> bool:
        span = self._span
        _current_span.reset(self._token)
        span.finish(status="error" if exc_type else "ok")
        if exc is not None:
            span.attributes.setdefault("error", repr(exc))
        _client.get_client().log_event(span.model_dump())
        return False

    async def __aenter__(self) -> RagEvent:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return self.__exit__(exc_type, exc, tb)

    # -- decorator --------------------------------------------------------
    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with _TraceHandle(self._name or fn.__name__, self._stage, dict(self._attributes)):
                return fn(*args, **kwargs)

        return wrapper


def trace(name: Any = None, stage: Optional[str] = None, **attributes):
    """``with ragobserve.trace("retrieval"): ...`` or ``@ragobserve.trace``.

    If ``name`` matches a lifecycle stage (retrieval, reranking, generation,
    ...) the span is tagged with that stage automatically.
    """
    if callable(name):  # bare @ragobserve.trace
        fn = name
        return _TraceHandle(fn.__name__, stage or _infer_stage(fn.__name__), {})(fn)
    name = name or ""
    return _TraceHandle(name, stage or _infer_stage(name), attributes)


def _infer_stage(name: str) -> str:
    low = (name or "").lower()
    if low in STAGES:
        return low
    for s in STAGES:
        if s in low or low.startswith(s[:7]):
            return s
    aliases = {
        "retrieve": Stage.RETRIEVAL, "search": Stage.RETRIEVAL,
        "rerank": Stage.RERANKING, "embed": Stage.EMBEDDING,
        "chunk": Stage.CHUNKING, "split": Stage.CHUNKING,
        "generate": Stage.GENERATION, "llm": Stage.GENERATION,
        "ingest": Stage.INGESTION, "context": Stage.CONTEXT_ASSEMBLY,
        "query": Stage.OTHER,
    }
    for key, stage in aliases.items():
        if key in low:
            return stage.value
    return Stage.OTHER.value


# ---------------------------------------------------------------- loggers

def _log(stage: str, name: str, attributes: Dict[str, Any], duration_ms: Optional[float] = None) -> RagEvent:
    parent = _current_span.get()
    now = time.time()
    ev = RagEvent(
        project=_client.get_project(),
        name=name or stage,
        stage=stage,
        start_time=now if duration_ms is None else now - duration_ms / 1000.0,
        end_time=now,
        duration_ms=duration_ms or 0.0,
        attributes=attributes,
    )
    if parent is not None:
        ev.trace_id = parent.trace_id
        ev.parent_span_id = parent.span_id
    _client.get_client().log_event(ev.model_dump())
    return ev


def log_ingestion(source: Optional[str] = None, count: Optional[int] = None,
                  sources: Optional[List[str]] = None, **extra) -> RagEvent:
    attrs = {"source": source, "count": count, "sources": sources, **extra}
    return _log(Stage.INGESTION.value, "ingestion", attrs)


def log_chunks(chunks: List[Any], strategy: Optional[str] = None, chunk_size: Optional[int] = None,
               overlap: Optional[int] = None, **extra) -> RagEvent:
    attrs = {"chunks": [normalize_result(c) for c in chunks], "strategy": strategy,
             "chunk_size": chunk_size, "overlap": overlap, **extra}
    return _log(Stage.CHUNKING.value, "chunking", attrs)


def log_embedding(model: str, input_count: int = 0, dimensions: Optional[int] = None,
                  cost: Optional[float] = None, duration_ms: Optional[float] = None, **extra) -> RagEvent:
    attrs = {"model": model, "input_count": input_count, "dimensions": dimensions, "cost": cost, **extra}
    return _log(Stage.EMBEDDING.value, "embedding", attrs, duration_ms)


def log_retrieval(query: str, results: List[Any], top_k: Optional[int] = None,
                  retriever: Optional[str] = None, duration_ms: Optional[float] = None, **extra) -> RagEvent:
    norm = [normalize_result(r) for r in results]
    for i, r in enumerate(norm):
        r.setdefault("rank", i + 1)
    attrs = {"query": query, "results": norm, "top_k": top_k or len(norm), "retriever": retriever, **extra}
    return _log(Stage.RETRIEVAL.value, "retrieval", attrs, duration_ms)


def log_fusion(results: List[Any], inputs: Optional[Dict[str, List[Any]]] = None,
               strategy: Optional[str] = None, **extra) -> RagEvent:
    attrs = {
        "results": [normalize_result(r) for r in results],
        "inputs": {k: [normalize_result(r) for r in v] for k, v in (inputs or {}).items()},
        "strategy": strategy, **extra,
    }
    return _log(Stage.FUSION.value, "fusion", attrs)


def log_rerank(before: List[Any], after: List[Any], model: Optional[str] = None,
               duration_ms: Optional[float] = None, **extra) -> RagEvent:
    attrs = {
        "before": [normalize_result(r) for r in before],
        "after": [normalize_result(r) for r in after],
        "model": model, **extra,
    }
    return _log(Stage.RERANKING.value, "reranking", attrs, duration_ms)


def log_context(final_prompt: str, system_prompt: Optional[str] = None, query: Optional[str] = None,
                chunks: Optional[List[Any]] = None, context_window: Optional[int] = None, **extra) -> RagEvent:
    from .events import estimate_tokens

    attrs = {
        "final_prompt": final_prompt, "system_prompt": system_prompt, "query": query,
        "chunks": [normalize_result(c) for c in (chunks or [])],
        "token_count": estimate_tokens(final_prompt), "context_window": context_window, **extra,
    }
    return _log(Stage.CONTEXT_ASSEMBLY.value, "context_assembly", attrs)


def log_generation(model: str, prompt: Optional[str] = None, response: Optional[str] = None,
                   input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
                   cost: Optional[float] = None, duration_ms: Optional[float] = None, **extra) -> RagEvent:
    attrs = {"model": model, "prompt": prompt, "response": response, "input_tokens": input_tokens,
             "output_tokens": output_tokens, "cost": cost, **extra}
    return _log(Stage.GENERATION.value, "generation", attrs, duration_ms)


def log_ground_truth(relevant_chunk_ids: List[str], trace_id: Optional[str] = None) -> None:
    tid = trace_id or current_trace_id()
    if tid is None:
        raise ValueError("log_ground_truth needs an active trace or an explicit trace_id")
    _client.get_client().log_ground_truth(tid, _client.get_project(), relevant_chunk_ids)
