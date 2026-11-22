"""OTLP/HTTP trace ingest: turn OpenTelemetry spans into RAGObserve events, so
apps already instrumented with OpenInference (Arize/Phoenix), OpenLLMetry
(Traceloop) or the GenAI semantic conventions need no code changes.

Point any OTLP exporter at ``http://<host>:5601/v1/traces`` with header
``Authorization: Bearer <key>``. JSON always works; protobuf (the default of
the Python/JS exporters) needs ``opentelemetry-proto`` on the server
(``pip install ragobserve[otel]``).

The project comes from the ``X-Ragobserve-Project`` header, else the
resource's ``service.name``, else ``default``.
"""
from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, List, Optional

_OI_KIND = {  # openinference.span.kind
    "RETRIEVER": "retrieval", "RERANKER": "reranking", "LLM": "generation",
    "EMBEDDING": "embedding",
}
_GENAI_OP = {  # gen_ai.operation.name
    "chat": "generation", "text_completion": "generation",
    "generate_content": "generation", "embeddings": "embedding",
}


def _value(v: Dict[str, Any]) -> Any:
    """Decode one OTLP AnyValue."""
    if not isinstance(v, dict):
        return v
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        return int(v["intValue"])  # int64 is a JSON string in OTLP
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "arrayValue" in v:
        return [_value(x) for x in (v["arrayValue"] or {}).get("values", [])]
    if "kvlistValue" in v:
        return _attrs((v["kvlistValue"] or {}).get("values", []))
    return None


def _attrs(kvs: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    return {kv["key"]: _value(kv.get("value") or {}) for kv in kvs or [] if "key" in kv}


def _indexed(a: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
    """``prefix.0.x``, ``prefix.1.x`` flattened keys -> ``[{x: ..}, {x: ..}]``."""
    rx = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.(.+)$")
    rows: Dict[int, Dict[str, Any]] = {}
    for k, v in a.items():
        m = rx.match(k)
        if m:
            rows.setdefault(int(m.group(1)), {})[m.group(2)] = v
    return [rows[i] for i in sorted(rows)]


def _docs(a: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
    out = []
    for i, d in enumerate(_indexed(a, prefix)):
        meta = d.get("document.metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except ValueError:
                meta = {"raw": meta}
        doc = {"text": d.get("document.content") or "", "score": d.get("document.score"),
               "rank": i + 1, "metadata": meta if isinstance(meta, dict) else {}}
        if d.get("document.id") is not None:
            doc["id"] = str(d["document.id"])
        out.append(doc)
    return out


def _first(a: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if a.get(k) not in (None, ""):
            return a[k]
    return None


def _messages(a: Dict[str, Any], prefix: str, content_key: str) -> Optional[str]:
    parts = [m.get(content_key) for m in _indexed(a, prefix) if m.get(content_key)]
    return "\n\n".join(str(p) for p in parts) or None


def _stage(name: str, a: Dict[str, Any]) -> str:
    kind = str(a.get("openinference.span.kind") or "").upper()
    if kind in _OI_KIND:
        return _OI_KIND[kind]
    if kind or a.get("traceloop.span.kind"):
        # CHAIN / AGENT / TOOL / workflow / task: a wrapper, whatever its name says.
        # Guessing from names like "RetrievalQA" would fake an empty retrieval.
        return "other"
    op = str(a.get("gen_ai.operation.name") or "").lower()
    if op in _GENAI_OP:
        return _GENAI_OP[op]
    if a.get("llm.request.type") in ("chat", "completion") or "gen_ai.system" in a:
        return "generation"
    if a.get("llm.request.type") == "embedding":
        return "embedding"
    if a.get("db.system") and any(k.startswith(("db.query", "db.vector")) for k in a):
        return "retrieval"
    from ..tracing import _infer_stage
    return _infer_stage(name)


def span_to_event(span: Dict[str, Any], project: str) -> Dict[str, Any]:
    a = _attrs(span.get("attributes"))
    name = span.get("name") or ""
    stage = _stage(name, a)
    start = int(span.get("startTimeUnixNano") or 0) / 1e9
    end = int(span.get("endTimeUnixNano") or 0) / 1e9 or start
    status = span.get("status") or {}
    is_error = status.get("code") in (2, "STATUS_CODE_ERROR")

    query = _first(a, "reranker.query", "input.value") if stage in ("retrieval", "reranking") else None
    attrs: Dict[str, Any] = {}
    if stage == "retrieval":
        attrs = {"query": query, "results": _docs(a, "retrieval.documents"),
                 "retriever": _first(a, "db.system") or name}
    elif stage == "reranking":
        attrs = {"query": query, "before": _docs(a, "reranker.input_documents"),
                 "after": _docs(a, "reranker.output_documents"),
                 "model": a.get("reranker.model_name")}
    elif stage == "generation":
        attrs = {
            "model": _first(a, "llm.model_name", "gen_ai.response.model", "gen_ai.request.model"),
            "provider": _first(a, "llm.provider", "gen_ai.system"),
            "prompt": _first(a, "input.value") or _messages(a, "llm.input_messages", "message.content")
                      or _messages(a, "gen_ai.prompt", "content"),
            "response": _first(a, "output.value") or _messages(a, "llm.output_messages", "message.content")
                        or _messages(a, "gen_ai.completion", "content"),
            "input_tokens": _first(a, "llm.token_count.prompt", "gen_ai.usage.input_tokens",
                                   "gen_ai.usage.prompt_tokens"),
            "output_tokens": _first(a, "llm.token_count.completion", "gen_ai.usage.output_tokens",
                                    "gen_ai.usage.completion_tokens"),
        }
    elif stage == "embedding":
        attrs = {"model": _first(a, "embedding.model_name", "gen_ai.request.model")}
    elif not span.get("parentSpanId") and a.get("input.value"):
        attrs = {"query": a["input.value"]}  # root chain span carries the user's question
    attrs = {k: v for k, v in attrs.items() if v is not None}
    if is_error and status.get("message"):
        attrs["error"] = status["message"]
    attrs["otel"] = {"name": name, "attributes": a}

    return {
        "event_id": f"{span.get('traceId', '')}{span.get('spanId', '')}",
        "trace_id": span.get("traceId"),
        "span_id": span.get("spanId"),
        "parent_span_id": span.get("parentSpanId") or None,
        "project": project,
        "stage": stage,
        "name": name or stage,
        "start_time": start,
        "end_time": end,
        "duration_ms": (end - start) * 1000.0,
        "status": "error" if is_error else "ok",
        "attributes": attrs,
    }


def otlp_to_events(payload: Dict[str, Any], project: Optional[str] = None) -> List[Dict[str, Any]]:
    """An OTLP ``ExportTraceServiceRequest`` (JSON shape) -> RAGObserve events."""
    events = []
    for rs in payload.get("resourceSpans") or []:
        res = _attrs((rs.get("resource") or {}).get("attributes"))
        proj = project or res.get("service.name") or "default"
        for ss in rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []:
            for span in ss.get("spans") or []:
                events.append(span_to_event(span, proj))
    return events


def decode_protobuf(body: bytes) -> Dict[str, Any]:
    """Protobuf request -> the JSON shape. Raises ImportError without opentelemetry-proto."""
    from google.protobuf.json_format import MessageToDict
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    req = ExportTraceServiceRequest()
    req.ParseFromString(body)
    d = MessageToDict(req)
    # MessageToDict base64-encodes bytes fields; OTLP/JSON uses hex for ids
    for rs in d.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for span in ss.get("spans", []):
                for k in ("traceId", "spanId", "parentSpanId"):
                    if span.get(k):
                        span[k] = base64.b64decode(span[k]).hex()
    return d
