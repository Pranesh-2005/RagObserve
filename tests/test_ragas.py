"""ragobserve.ragas: trace -> Ragas sample mapping, CLI export, and the eval
backend (Ragas itself faked, so this runs without ragas installed)."""
from __future__ import annotations

import json
import sys
import types
from unittest.mock import patch

import pytest

from ragobserve.ragas import evaluate_trace, to_ragas_rows, trace_to_sample


def _trace(with_context=True):
    events = [
        {"stage": "retrieval", "attributes": {"query": "notice?", "results": [
            {"id": "c1", "text": "Notice is 90 days."}, {"id": "c2", "text": "Unrelated."},
            {"id": "img:1", "text": "", "metadata": {"modality": "image"}}]}},
        {"stage": "generation", "attributes": {"response": "90 days.", "reference": "90 days"}},
    ]
    if with_context:
        events.insert(1, {"stage": "context_assembly", "attributes": {"chunks": [
            {"id": "c1", "text": "Notice is 90 days."}]}})
    return {"trace": {"query": "notice?", "trace_id": "t-1"}, "events": events, "ground_truth": ["c1"]}


def test_sample_prefers_assembled_context():
    s = trace_to_sample(_trace())
    assert s.pop("trace_id") == "t-1"
    assert s == {
        "user_input": "notice?", "response": "90 days.",
        "retrieved_contexts": ["Notice is 90 days."], "retrieved_context_ids": ["c1"],
        "reference": "90 days", "reference_context_ids": ["c1"],
    }


def test_sample_falls_back_to_retrieval_and_skips_empty_text():
    s = trace_to_sample(_trace(with_context=False))
    assert s["retrieved_contexts"] == ["Notice is 90 days.", "Unrelated."]
    assert s["retrieved_context_ids"] == ["c1", "c2", "img:1"]


def test_rows_skip_missing_traces():
    assert len(to_ragas_rows([_trace(), None])) == 1


def test_evaluate_maps_ragas_metrics_to_dashboard_keys():
    seen = {}

    class _Ds:
        @staticmethod
        def from_list(rows):
            seen["rows"] = rows
            return "DS"

    class _Result:
        scores = [{"faithfulness": 1.0, "answer_relevancy": float("nan"), "context_recall": 0.5}]

    def fake_evaluate(ds, metrics, llm, embeddings, show_progress):
        seen["ds"] = ds
        return _Result()

    ragas_mod = types.SimpleNamespace(EvaluationDataset=_Ds, evaluate=fake_evaluate)
    metrics_mod = types.SimpleNamespace(Faithfulness=lambda: "F", ResponseRelevancy=lambda: "R")
    with patch.dict(sys.modules, {"ragas": ragas_mod, "ragas.metrics": metrics_mod}):
        out = evaluate_trace(_trace())

    assert seen["ds"] == "DS" and seen["rows"][0]["user_input"] == "notice?"
    assert out["faithfulness"]["score"] == 1.0
    assert out["answer_relevance"]["score"] is None  # NaN -> None
    assert out["context_recall"]["score"] == 0.5


def test_evaluate_without_ragas_says_how_to_install():
    with patch.dict(sys.modules, {"ragas": None}):
        with pytest.raises(RuntimeError, match=r"ragobserve\[ragas\]"):
            evaluate_trace(_trace())


def test_cli_export_ragas(tmp_path):
    import time
    import uuid

    from ragobserve.cli import main
    from ragobserve.server.db import Store

    db = str(tmp_path / "r.db")
    s = Store(db)
    tid = uuid.uuid4().hex
    base = {"trace_id": tid, "project": "rp", "start_time": time.time(), "end_time": time.time()}
    s.ingest_events([
        {**base, "event_id": "e1", "stage": "retrieval",
         "attributes": {"query": "q?", "results": [{"id": "c1", "text": "ctx"}]}},
        {**base, "event_id": "e2", "stage": "generation", "attributes": {"response": "ans"}},
    ])
    s.close()

    out = tmp_path / "ragas.jsonl"
    assert main(["export", "--project", "rp", "--format", "ragas", "-o", str(out),
                 "--backend-store-uri", db]) == 0
    row = json.loads(out.read_text().splitlines()[0])
    assert row["user_input"] == "q?" and row["response"] == "ans"
    assert row["retrieved_contexts"] == ["ctx"]
