"""OTel ingest, image cost, replay with images, grounding, triage, compare,
golden-set regression, Ragas import."""
from __future__ import annotations

import json
import textwrap
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ragobserve.server import metrics as M
from ragobserve.server import pricing
from ragobserve.server.app import create_app
from ragobserve.server.db import Store

KEY = "feat-key"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGOBSERVE_API_KEY", KEY)
    app = create_app(str(tmp_path / "f.db"))
    with TestClient(app, headers={"Authorization": f"Bearer {KEY}"}) as c:
        yield c
    app.state.store.close()


def _kv(k, v):
    if isinstance(v, bool):
        return {"key": k, "value": {"boolValue": v}}
    if isinstance(v, int):
        return {"key": k, "value": {"intValue": str(v)}}
    if isinstance(v, float):
        return {"key": k, "value": {"doubleValue": v}}
    return {"key": k, "value": {"stringValue": v}}


def _span(sid, name, attrs, parent=None, t0=1_700_000_000_000_000_000, ms=10, error=False):
    s = {"traceId": "ab" * 16, "spanId": sid, "name": name,
         "startTimeUnixNano": str(t0), "endTimeUnixNano": str(t0 + ms * 1_000_000),
         "attributes": [_kv(k, v) for k, v in attrs.items()]}
    if parent:
        s["parentSpanId"] = parent
    if error:
        s["status"] = {"code": 2, "message": "boom"}
    return s


def _otlp(spans, service="otel-app"):
    return {"resourceSpans": [{"resource": {"attributes": [_kv("service.name", service)]},
                               "scopeSpans": [{"spans": spans}]}]}


OPENINFERENCE = [
    _span("01" * 8, "RetrievalQA", {"openinference.span.kind": "CHAIN",
                                    "input.value": "notice period?"}, ms=50),
    _span("02" * 8, "retrieve", {
        "openinference.span.kind": "RETRIEVER", "input.value": "notice period?",
        "retrieval.documents.0.document.id": "c1",
        "retrieval.documents.0.document.content": "Notice is 90 days.",
        "retrieval.documents.0.document.score": 0.9,
        "retrieval.documents.1.document.id": "c2",
        "retrieval.documents.1.document.content": "Other.",
        "retrieval.documents.1.document.score": 0.4,
    }, parent="01" * 8),
    _span("03" * 8, "ChatOpenAI", {
        "openinference.span.kind": "LLM", "llm.model_name": "gpt-4o-mini",
        "llm.input_messages.0.message.content": "Context... Q: notice period?",
        "llm.output_messages.0.message.content": "90 days.",
        "llm.token_count.prompt": 120, "llm.token_count.completion": 4,
    }, parent="01" * 8),
]


# ------------------------------------------------------------------ 9. OTel

def test_otlp_json_openinference(client):
    r = client.post("/v1/traces", json=_otlp(OPENINFERENCE))
    assert r.status_code == 200
    tid = "ab" * 16
    t = client.get(f"/api/traces/{tid}").json()
    assert t["trace"]["project"] == "otel-app"
    assert t["trace"]["query"] == "notice period?"
    by_stage = {e["stage"]: e["attributes"] for e in t["events"]}
    assert set(by_stage) == {"other", "retrieval", "generation"}  # CHAIN not misread as retrieval
    assert [d["id"] for d in by_stage["retrieval"]["results"]] == ["c1", "c2"]
    assert by_stage["retrieval"]["results"][0]["score"] == 0.9
    gen = by_stage["generation"]
    assert gen["model"] == "gpt-4o-mini" and gen["response"] == "90 days."
    assert gen["input_tokens"] == 120 and gen["cost"] > 0 and gen["cost_estimated"]


def test_otlp_genai_semconv_and_project_header(client):
    spans = [_span("04" * 8, "chat gpt-4o", {
        "gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4o",
        "gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 2,
        "gen_ai.prompt.0.content": "hi", "gen_ai.completion.0.content": "hello",
    }, error=True)]
    r = client.post("/v1/traces", json=_otlp(spans), headers={"X-Ragobserve-Project": "forced"})
    assert r.status_code == 200
    t = client.get(f"/api/traces/{'ab' * 16}").json()
    ev = t["events"][0]
    assert t["trace"]["project"] == "forced"
    assert ev["stage"] == "generation" and ev["status"] == "error"
    assert ev["attributes"]["prompt"] == "hi" and ev["attributes"]["error"] == "boom"


def test_otlp_requires_key(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGOBSERVE_API_KEY", KEY)
    app = create_app(str(tmp_path / "u.db"))
    with TestClient(app) as c:
        assert c.post("/v1/traces", json=_otlp(OPENINFERENCE)).status_code == 401
    app.state.store.close()


def test_otlp_protobuf_from_real_sdk(client):
    pytest.importorskip("opentelemetry.proto")
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    mem = InMemorySpanExporter()
    tp = TracerProvider(resource=Resource.create({"service.name": "pb-app"}))
    tp.add_span_processor(SimpleSpanProcessor(mem))
    tracer = tp.get_tracer("t")
    with tracer.start_as_current_span("rag") as root:
        root.set_attribute("openinference.span.kind", "CHAIN")
        with tracer.start_as_current_span("retrieve") as s:
            s.set_attribute("openinference.span.kind", "RETRIEVER")
            s.set_attribute("input.value", "q?")
            s.set_attribute("retrieval.documents.0.document.id", "c9")
            s.set_attribute("retrieval.documents.0.document.content", "txt")
    body = encode_spans(mem.get_finished_spans()).SerializeToString()
    tid = format(root.get_span_context().trace_id, "032x")

    r = client.post("/v1/traces", content=body, headers={"Content-Type": "application/x-protobuf"})
    assert r.status_code == 200, r.text
    t = client.get(f"/api/traces/{tid}").json()  # ids decoded to the same hex the SDK uses
    assert t["trace"]["project"] == "pb-app"
    ret = [e for e in t["events"] if e["stage"] == "retrieval"][0]
    assert ret["attributes"]["results"][0]["id"] == "c9"
    child = [e for e in t["events"] if e["stage"] == "retrieval"][0]
    assert child["parent_span_id"] == format(root.get_span_context().span_id, "016x")


# ------------------------------------------------------------ 10a. image cost

def test_image_tokens_match_vendor_examples():
    assert pricing.image_tokens("gpt-4o", 1024, 1024) == 765        # OpenAI docs example
    assert pricing.image_tokens("gpt-4o", 1024, 1024, "low") == 85
    assert pricing.image_tokens("claude-sonnet-4-5", 1000, 1000) == 1334  # ceil(1e6/750)
    assert pricing.image_tokens("claude-sonnet-4-5", 3136, 3136) == pricing.image_tokens(
        "claude-sonnet-4-5", 1568, 1568)                               # long edge capped
    assert pricing.image_tokens("gemini-2.0-flash", 300, 300) == 258


def test_backfill_adds_image_tokens_to_cost():
    text_only = pricing.backfill_generation({"model": "gpt-4o", "input_tokens": 1000})
    with_img = pricing.backfill_generation({"model": "gpt-4o", "input_tokens": 1000,
                                            "images": [{"width": 1024, "height": 1024}]})
    assert with_img["image_tokens"] == 765
    assert with_img["cost"] == pytest.approx(pricing.estimate_cost("gpt-4o", 1765, None))
    assert with_img["cost"] > text_only["cost"]
    assert pricing.backfill_generation({"model": "gpt-4o", "cost": 0.5})["cost"] == 0.5


# -------------------------------------------------------- 10b. replay images

def test_replay_sends_trace_images(client, tmp_path, monkeypatch):
    from ragobserve.server.ratelimit import limiter

    # the limiter is process-global; other tests already spent /api/generate's 10/min
    monkeypatch.setattr(limiter, "enabled", False)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    img = tmp_path / "p.png"
    img.write_bytes(b"\x89PNG fake")
    tid = uuid.uuid4().hex
    client.post("/api/events", json={"events": [{
        "event_id": "e1", "trace_id": tid, "project": "mm", "stage": "context_assembly",
        "start_time": time.time(), "attributes": {"final_prompt": "Describe.", "chunks": [
            {"id": "i1", "text": "", "metadata": {"modality": "image", "path": str(img)}},
            {"id": "i2", "text": "", "metadata": {"modality": "image", "image_b64": "QUJD", "mime": "image/jpeg"}},
            {"id": "t1", "text": "plain chunk"}]}}]})
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"choices": [{"message": {"content": "a chart"}}], "model": "gpt-4o-mini"}
    with patch("httpx.post", return_value=resp) as post:
        r = client.post("/api/generate", json={"provider": "openai", "trace_id": tid})
    assert r.status_code == 200 and r.json()["image_count"] == 2
    content = post.call_args.kwargs["json"]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "Describe."}
    urls = [p["image_url"]["url"] for p in content[1:]]
    assert urls[0].startswith("data:image/png;base64,") and urls[1] == "data:image/jpeg;base64,QUJD"


def test_anthropic_image_parts():
    from ragobserve.server import llm

    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"content": [{"type": "text", "text": "ok"}], "usage": {}}
    with patch("httpx.post", return_value=resp) as post:
        llm.generate("anthropic", "", "Q", api_key="k", images=[("image/png", "QUJD")])
    parts = post.call_args.kwargs["json"]["messages"][0]["content"]
    assert parts[0]["type"] == "image" and parts[0]["source"]["data"] == "QUJD"
    assert parts[-1] == {"type": "text", "text": "Q"}


# ------------------------------------------------------------ 10c. grounding

def test_log_grounding_ratio(local_client):
    import ragobserve

    with ragobserve.trace("q") as root:
        ragobserve.log_grounding([
            {"sentence": "90 days.", "supported": True, "chunk_id": "c1"},
            {"sentence": "Signed 2020.", "supported": False},
            {"sentence": "unjudged"},
        ], method="nli")
    ev = [e for e in local_client.store.get_trace(root.trace_id)["events"] if e["stage"] == "grounding"][0]
    assert ev["attributes"]["supported_ratio"] == 0.5
    assert ev["attributes"]["claims"][0]["chunk_ids"] == ["c1"]


# --------------------------------------------------------------- 6. triage

def _t(events, gt=None, status="ok"):
    return {"trace": {"status": status}, "events": events, "ground_truth": gt}


R = {"stage": "retrieval", "attributes": {"results": [{"id": "c1"}, {"id": "c2"}]}}


@pytest.mark.parametrize("trace,scores,expected", [
    (_t([R], status="error"), {}, "error"),
    (_t([{"stage": "retrieval", "attributes": {"results": []}}]), {}, "empty_retrieval"),
    (_t([R], gt=["c9"]), {}, "retrieval_miss"),
    (_t([R, {"stage": "reranking", "attributes": {"after": [{"id": "c2"}]}}], gt=["c1"]), {}, "rerank_drop"),
    (_t([R, {"stage": "context_assembly", "attributes": {"chunks": [{"id": "c2"}]}}], gt=["c1"]), {}, "context_drop"),
    (_t([R, {"stage": "grounding", "attributes": {"supported_ratio": 0.2}}]), {}, "unsupported_claims"),
    (_t([R]), {"faithfulness": 0.2}, "unfaithful"),
    (_t([R]), {"faithfulness": 0.9, "answer_relevance": 0.1}, "off_topic"),
    (_t([R], gt=["c1"]), {}, "ok"),
    (_t([R]), {}, "unknown"),
])
def test_triage(trace, scores, expected):
    assert M.triage(trace, scores)["verdict"] == expected


def _ingest(client, project, gt, results):
    tid = uuid.uuid4().hex
    client.post("/api/events", json={"events": [
        {"event_id": uuid.uuid4().hex, "trace_id": tid, "project": project, "stage": "retrieval",
         "start_time": time.time(), "end_time": time.time() + 0.1,
         "attributes": {"query": "q", "results": [{"id": i} for i in results]}}]})
    client.post("/api/ground-truth", json={"trace_id": tid, "project": project, "relevant_chunk_ids": gt})
    return tid


def test_triage_and_compare_endpoints(client):
    _ingest(client, "A", ["c1"], ["c9"])          # miss
    _ingest(client, "B", ["c1"], ["c1", "c2"])    # hit
    tri = client.get("/api/triage", params={"project": "A"}).json()
    assert tri["counts"] == {"retrieval_miss": 1}
    tid = tri["traces"][0]["trace_id"]
    assert client.get(f"/api/traces/{tid}").json()["derived"]["triage"]["verdict"] == "retrieval_miss"

    cmp = client.get("/api/compare", params={"a": "A", "b": "B", "k": 5}).json()
    assert cmp["a"]["retrieval"]["recall_at_k"] == 0.0
    assert cmp["b"]["retrieval"]["recall_at_k"] == 1.0
    assert cmp["b"]["triage"] == {"ok": 1}
    assert cmp["a"]["latency_p50_ms"] == pytest.approx(100, rel=0.1)
    assert client.get("/compare").status_code == 200


# --------------------------------------------------------------- 7. golden

PIPELINE = textwrap.dedent("""
    import ragobserve

    def answer(q):
        hits = [{"id": "c1", "text": "Notice is 90 days."}] if "notice" in q else [{"id": "zz", "text": "?"}]
        ragobserve.log_retrieval(q, hits)
        return "90 days." if hits[0]["id"] == "c1" else "no idea"
""")


@pytest.fixture
def golden_env(tmp_path, monkeypatch):
    (tmp_path / "golden_pipe.py").write_text(PIPELINE)
    (tmp_path / "golden.jsonl").write_text(
        json.dumps({"question": "notice period?", "relevant_chunk_ids": ["c1"]}) + "\n"
        + json.dumps({"question": "renewal terms?", "relevant_chunk_ids": ["c7"]}) + "\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_golden_cli_pass_and_fail(golden_env, capsys):
    from ragobserve.cli import main

    db = str(golden_env / "g.db")
    base = ["test", "--golden", "golden.jsonl", "--pipeline", "golden_pipe:answer",
            "--backend-store-uri", db]
    assert main(base + ["--project", "run1", "--min-recall", "0.5"]) == 0
    assert "PASS" in capsys.readouterr().out
    assert main(base + ["--project", "run2", "--min-recall", "0.9"]) == 1
    assert "recall: 0.500 < 0.9" in capsys.readouterr().out

    s = Store(db)
    gen = [e for t in s.list_traces("run1") for e in s.get_trace(t["trace_id"])["events"]
           if e["stage"] == "generation"]
    assert sorted(g["attributes"]["response"] for g in gen) == ["90 days.", "no idea"]
    s.close()


def test_golden_missing_metric_fails_gate():
    from ragobserve.golden import check

    report = {"faithfulness": None, "recall": 1.0, "errors": []}
    assert check(report, {"faithfulness": 0.7, "recall": 0.5}) == ["faithfulness: no data (min 0.7)"]


def test_golden_judge_scores(golden_env):
    from ragobserve import golden

    store = Store(str(golden_env / "j.db"))
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"choices": [{"message": {"content": '{"score": 0.8, "reason": "r"}'}}]}
    with patch("httpx.post", return_value=resp):
        rep = golden.run(golden.load_pipeline("golden_pipe:answer"),
                         golden.load_golden("golden.jsonl"), store, "j",
                         judge={"provider": "groq", "api_key": "k"})
    assert rep["answer_relevance"] == pytest.approx(0.8)
    assert rep["questions"] == 2 and not rep["errors"]
    store.close()


# --------------------------------------------------------- 4. ragas import

def test_import_ragas_csv(tmp_path, capsys):
    from ragobserve.cli import main

    db = str(tmp_path / "i.db")
    s = Store(db)
    t1, t2 = uuid.uuid4().hex, uuid.uuid4().hex
    for tid, q in ((t1, "first q"), (t2, "second q")):
        s.ingest_events([{"event_id": uuid.uuid4().hex, "trace_id": tid, "project": "rp",
                          "stage": "retrieval", "start_time": time.time(), "attributes": {"query": q}}])
    s.close()
    csv_path = tmp_path / "res.csv"
    csv_path.write_text(
        ",trace_id,user_input,retrieved_contexts,faithfulness,answer_relevancy,context_recall\n"
        f"0,{t1},first q,['x'],0.9,0.8,1.0\n"
        "1,,second q,['y'],0.4,nan,0.5\n"
        "2,,never logged,[],0.1,0.1,0.1\n")
    assert main(["import-ragas", str(csv_path), "--project", "rp", "--backend-store-uri", db]) == 0
    assert "matched 2 rows" in capsys.readouterr().out

    s = Store(db)
    a = {r["metric"]: r["score"] for r in s.get_eval_scores(t1)}
    b = {r["metric"]: r["score"] for r in s.get_eval_scores(t2)}
    s.close()
    assert a == {"faithfulness": 0.9, "answer_relevance": 0.8, "context_recall": 1.0}
    assert b["faithfulness"] == 0.4 and "answer_relevance" not in b  # NaN never overwrites


def test_golden_scores_retrieval_even_when_generation_raises(tmp_path, monkeypatch):
    """Real-app finding: the LLM 404'd after retrieval, and recall showed 'no data'
    because ground truth was logged after the pipeline returned."""
    import ragobserve
    from ragobserve import golden

    def pipeline(q):
        ragobserve.log_retrieval(q, [{"id": "c1", "text": "t"}])
        raise RuntimeError("model_not_found")

    store = Store(str(tmp_path / "e.db"))
    rep = golden.run(pipeline, [{"question": "q", "relevant_chunk_ids": ["c1"], "reference": "gold"}],
                     store, "e")
    assert rep["recall"] == 1.0 and len(rep["errors"]) == 1
    from ragobserve.ragas import trace_to_sample
    tid = store.list_traces("e")[0]["trace_id"]
    assert trace_to_sample(store.get_trace(tid))["reference"] == "gold"
    store.close()


def test_triage_matches_mixed_id_schemes():
    """Real-app finding (examples/test_rag.py): int ids on retrieval, none on
    rerank/context; ground truth as text hashes. Was 'retrieval_miss' at recall 1.0."""
    from ragobserve.events import content_hash

    doc = "BM25 is lexical."
    t = _t([{"stage": "retrieval", "attributes": {"results": [{"id": 1, "text": doc}]}},
            {"stage": "reranking", "attributes": {"after": [{"text": doc}]}},
            {"stage": "context_assembly", "attributes": {"chunks": [{"text": doc}]}}],
           gt=[content_hash(doc)])
    assert M.triage(t)["verdict"] == "ok"
    assert M.triage({**t, "ground_truth": [1]})["verdict"] == "ok"  # int ground truth too


def test_root_span_query_wins_over_child_rewrite(tmp_path):
    s = Store(str(tmp_path / "q.db"))
    base = {"trace_id": "t", "project": "p", "start_time": time.time()}
    s.ingest_events([
        {**base, "event_id": "c", "span_id": "c", "parent_span_id": "r", "stage": "retrieval",
         "attributes": {"query": "rewritten"}},
        {**base, "event_id": "r", "span_id": "r", "stage": "other", "attributes": {"query": "user question"}},
    ])
    assert s.get_trace("t")["trace"]["query"] == "user question"
    s.close()
