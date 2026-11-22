"""Ragas bridge: turn RAGObserve traces into Ragas samples, and score them with
Ragas metrics so the scores land in the same dashboard as the built-in judge.

Export only needs RAGObserve (rows are plain dicts). Scoring needs
``pip install ragobserve[ragas]``; with no ``llm=``/``embeddings=`` passed,
Ragas uses its own default (OpenAI, ``OPENAI_API_KEY``).

Usage::

    from ragobserve.ragas import to_ragas_rows, to_ragas_dataset, evaluate_traces

    rows = to_ragas_rows([store.get_trace(tid) for tid in ids])   # list[dict]
    ds = to_ragas_dataset(traces)                                 # ragas.EvaluationDataset
    scores = evaluate_traces(traces)                              # per-trace {metric: {score, reason}}
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .events import context_items

# Ragas metric name -> the metric key RAGObserve already stores and renders, so
# Ragas scores fill the existing faithfulness / relevance columns.
_METRIC_KEYS = {"faithfulness": "faithfulness", "answer_relevancy": "answer_relevance"}


def trace_to_sample(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """One trace (from ``get_trace``) -> one Ragas single-turn sample dict."""
    events = trace_data.get("events") or []
    query = (trace_data.get("trace") or {}).get("query") or ""
    response, reference = "", None
    for ev in events:
        a = ev.get("attributes") or {}
        query = query or a.get("query") or ""
        if ev.get("stage") == "generation" and not response:
            response = a.get("response") or ""
        # log_generation(..., reference="gold answer") flows through **extra
        reference = reference or a.get("reference")

    items = context_items(events)
    sample: Dict[str, Any] = {
        "user_input": query,
        "response": response,
        "retrieved_contexts": [i["text"] for i in items if i.get("text")],
        "retrieved_context_ids": [str(i["id"]) for i in items if i.get("id")],
    }
    if reference:
        sample["reference"] = reference
    tid = (trace_data.get("trace") or {}).get("trace_id")
    if tid:
        sample["trace_id"] = tid  # Ragas ignores unknown keys; import_results matches on it
    if trace_data.get("ground_truth"):
        sample["reference_context_ids"] = [str(x) for x in trace_data["ground_truth"]]
    return sample


def to_ragas_rows(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [trace_to_sample(t) for t in traces if t]


def to_ragas_dataset(traces: List[Dict[str, Any]]):
    try:
        from ragas import EvaluationDataset
    except ImportError as e:
        raise RuntimeError("Ragas not installed: pip install 'ragobserve[ragas]'") from e
    return EvaluationDataset.from_list(to_ragas_rows(traces))


def _clean(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def evaluate_traces(traces: List[Dict[str, Any]], metrics: Optional[list] = None,
                    llm: Any = None, embeddings: Any = None) -> List[Dict[str, Dict[str, Any]]]:
    """Score traces with Ragas. Default metrics: Faithfulness + ResponseRelevancy.

    Returns one ``{metric: {"score": float|None, "reason": str}}`` per trace, in
    order: the same shape ``ragobserve.eval.evaluate_trace`` returns.
    """
    traces = [t for t in traces if t]
    if not traces:
        return []
    try:
        from ragas import evaluate
        from ragas.metrics import Faithfulness, ResponseRelevancy
    except ImportError as e:
        raise RuntimeError("Ragas not installed: pip install 'ragobserve[ragas]'") from e

    result = evaluate(
        to_ragas_dataset(traces),
        metrics=metrics or [Faithfulness(), ResponseRelevancy()],
        llm=llm, embeddings=embeddings, show_progress=False,
    )
    out = []
    for row in result.scores:
        out.append({
            _METRIC_KEYS.get(name, name): {"score": _clean(val), "reason": "ragas"}
            for name, val in row.items()
        })
    return out


def evaluate_trace(trace_data: Dict[str, Any], **kw: Any) -> Dict[str, Dict[str, Any]]:
    res = evaluate_traces([trace_data], **kw)
    return res[0] if res else {}


# Columns Ragas writes that are sample fields, not scores.
_SAMPLE_FIELDS = {"trace_id", "user_input", "response", "reference", "retrieved_contexts",
                  "reference_contexts", "retrieved_context_ids", "reference_context_ids",
                  "multi_responses", "rubrics", "persona_name", "query_style", "query_length"}


def read_results(path: str) -> List[Dict[str, Any]]:
    """A Ragas results file: ``result.to_pandas().to_csv(...)`` or JSONL."""
    import csv
    import json

    with open(path, encoding="utf-8", newline="") as f:
        if path.lower().endswith((".jsonl", ".ndjson", ".json")):
            return [json.loads(line) for line in f if line.strip()]
        return list(csv.DictReader(f))


def import_results(store: Any, project: str, rows: List[Dict[str, Any]],
                   model: str = "ragas-import") -> Dict[str, Any]:
    """Attach scores from a Ragas run done elsewhere onto their traces.

    Rows match by ``trace_id`` (present in ``export --format ragas`` output),
    else by ``user_input`` == the trace's query, newest trace first. Every
    numeric non-sample column is stored as a metric; ``answer_relevancy`` is
    stored as ``answer_relevance`` so it fills the dashboard column.
    """
    by_query: Dict[str, str] = {}
    for t in store.list_traces(project, 100_000):  # newest first, keep the first seen
        if t.get("query"):
            by_query.setdefault(t["query"].strip(), t["trace_id"])

    matched, unmatched, scores = 0, [], 0
    for row in rows:
        tid = (row.get("trace_id") or "").strip() or by_query.get((row.get("user_input") or "").strip())
        if not tid or store.get_trace(tid) is None:
            unmatched.append(row.get("user_input") or row.get("trace_id") or "?")
            continue
        matched += 1
        for col, val in row.items():
            if col in _SAMPLE_FIELDS or not col or col.startswith("Unnamed"):  # pandas index col
                continue
            score = _clean(val)
            if score is None:
                # text column, or a metric Ragas failed on (NaN): never overwrite
                # an existing score with nothing
                continue
            store.set_eval_score(tid, project, _METRIC_KEYS.get(col, col), score, "ragas", model)
            scores += 1
    return {"matched": matched, "unmatched": unmatched, "scores": scores}
