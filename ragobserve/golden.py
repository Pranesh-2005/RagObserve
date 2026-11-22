"""Golden-set regression: run a fixed question set through your instrumented
pipeline, score it, and fail when quality drops. Built for CI.

golden.jsonl, one question per line::

    {"question": "What is the notice period?", "relevant_chunk_ids": ["c12"], "reference": "90 days"}

Your pipeline is any ``fn(question) -> answer`` that calls ``ragobserve.log_*``
inside. Each question runs in its own trace in a fresh project
(``golden-YYYYmmdd-HHMMSS`` by default), so runs line up on the Compare page.

    ragobserve test --golden golden.jsonl --pipeline my_app.rag:answer --min-recall 0.8
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from .server import metrics as M

THRESHOLDS = ("recall", "precision", "mrr", "ndcg", "faithfulness", "answer_relevance")
_AGG_KEY = {"recall": "recall_at_k", "precision": "precision_at_k", "mrr": "mrr", "ndcg": "ndcg"}


def load_golden(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            q = row.get("question") or row.get("user_input")
            if not q:
                raise ValueError(f"{path}:{n}: needs a 'question' (or 'user_input') field")
            rows.append({**row, "question": q})
    return rows


def load_pipeline(spec: str) -> Callable[[str], Any]:
    """``package.module:function`` -> the callable. The cwd is importable."""
    mod, _, fn = spec.partition(":")
    if not mod or not fn:
        raise ValueError(f"--pipeline must look like module:function, got {spec!r}")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(mod), fn)


def run(pipeline: Callable[[str], Any], rows: List[Dict[str, Any]], store: Any, project: str,
        k: int = 10, judge: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run every golden row through ``pipeline`` and score the run.

    ``judge`` (``{"provider":..,"model":..,"api_key":..}``) also scores
    faithfulness / answer relevance with the LLM judge.
    """
    import ragobserve
    from . import client as _client

    prev = (_client._config.project, _client._config.client)
    ragobserve.init(project=project, store=store)
    trace_ids, errors = [], []
    try:
        for row in rows:
            try:
                # reference on the root span reaches Ragas export even when the
                # pipeline logs its own generation
                with ragobserve.trace("golden", query=row["question"],
                                      **({"reference": row["reference"]} if row.get("reference") else {})) as root:
                    trace_ids.append(root.trace_id)
                    # before the pipeline: a run that raises mid-way still has
                    # retrieval worth scoring
                    if row.get("relevant_chunk_ids"):
                        ragobserve.log_ground_truth(row["relevant_chunk_ids"])
                    answer = pipeline(row["question"])
                    if inspect.isawaitable(answer):
                        answer = asyncio.run(answer)
                    if isinstance(answer, str) and not _has_generation(store, root.trace_id):
                        ragobserve.log_generation(model="pipeline", response=answer)
            except Exception as e:  # one bad question shouldn't hide the rest of the run
                errors.append({"question": row["question"], "error": repr(e)})
    finally:
        _client._config.project, _client._config.client = prev

    ids = set(trace_ids)
    pairs = [p for p in store.traces_with_ground_truth(project) if p["trace_id"] in ids]
    agg = M.evaluate_traces(pairs, k)["aggregate"]
    report: Dict[str, Any] = {
        "project": project, "questions": len(rows), "errors": errors, "k": k,
        "recall": agg["recall_at_k"], "precision": agg["precision_at_k"],
        "mrr": agg["mrr"], "ndcg": agg["ndcg"], "faithfulness": None, "answer_relevance": None,
    }
    if judge:
        from .eval import evaluate_trace
        from .server import llm

        model = judge.get("model") or llm.default_model(judge.get("provider", "groq"))
        faith, rel = [], []
        for tid in trace_ids:
            detail = store.get_trace(tid)
            if detail is None:
                continue
            scores = evaluate_trace(detail, model=model, api_key=judge.get("api_key", ""),
                                    provider=judge.get("provider", "groq"))
            for metric, r in scores.items():
                store.set_eval_score(tid, project, metric, r.get("score"), r.get("reason", ""), model)
            faith.append(scores["faithfulness"]["score"])
            rel.append(scores["answer_relevance"]["score"])
        report["faithfulness"] = M._mean(faith)
        report["answer_relevance"] = M._mean(rel)
    return report


def _has_generation(store: Any, trace_id: str) -> bool:
    detail = store.get_trace(trace_id)
    return bool(detail) and any(e["stage"] == "generation" for e in detail["events"])


def check(report: Dict[str, Any], minimums: Dict[str, Optional[float]]) -> List[str]:
    """Threshold failures as messages. A metric with no data fails its threshold:
    a gate that silently passes on missing data isn't a gate."""
    fails = []
    for name, floor in minimums.items():
        if floor is None:
            continue
        got = report.get(name)
        if got is None:
            fails.append(f"{name}: no data (min {floor})")
        elif got < floor:
            fails.append(f"{name}: {got:.3f} < {floor}")
    if report["errors"]:
        fails.append(f"{len(report['errors'])} question(s) raised")
    return fails


def default_project() -> str:
    return time.strftime("golden-%Y%m%d-%H%M%S")
