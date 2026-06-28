"""Tests for ragobserve.eval — LLM-as-judge metrics (Groq mocked via httpx.post)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ragobserve.eval import evaluate_trace, score_answer_relevance, score_faithfulness


def _groq_ok(score: float, reason: str = "ok") -> MagicMock:
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {
        "choices": [{"message": {"content": json.dumps({"score": score, "reason": reason})}}]
    }
    return m


# ---------------------------------------------------------------------------
# score_faithfulness
# ---------------------------------------------------------------------------

def test_faithfulness_grounded():
    with patch("httpx.post", return_value=_groq_ok(0.95, "All claims supported by context.")):
        r = score_faithfulness(
            answer="The notice period is 90 days.",
            context=["Notice period is 90 days as per the contract."],
            api_key="fake-key",
        )
    assert r["score"] == pytest.approx(0.95)
    assert "supported" in r["reason"]


def test_faithfulness_empty_context():
    r = score_faithfulness("some answer", [], api_key="fake-key")
    assert r["score"] is None
    assert "no context" in r["reason"]


def test_faithfulness_empty_answer():
    r = score_faithfulness("", ["some context"], api_key="fake-key")
    assert r["score"] is None
    assert "no answer" in r["reason"]


def test_faithfulness_dict_chunks():
    with patch("httpx.post", return_value=_groq_ok(1.0, "perfect")):
        r = score_faithfulness(
            answer="X is Y.",
            context=[{"text": "X is Y.", "score": 0.9}],
            api_key="fake-key",
        )
    assert r["score"] == pytest.approx(1.0)


def test_faithfulness_no_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        score_faithfulness("answer", ["context"])


def test_faithfulness_malformed_llm_json():
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {"choices": [{"message": {"content": "I cannot score this."}}]}
    with patch("httpx.post", return_value=m):
        with pytest.raises(ValueError, match="non-JSON"):
            score_faithfulness("answer", ["ctx"], api_key="fake-key")


def test_faithfulness_nested_json_in_response():
    """LLM wraps answer in markdown or extra text — parser still extracts score."""
    content = 'Sure! Here is my evaluation:\n{"score": 0.75, "reason": "mostly grounded"}\nDone.'
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {"choices": [{"message": {"content": content}}]}
    with patch("httpx.post", return_value=m):
        r = score_faithfulness("answer text", ["ctx chunk"], api_key="fake-key")
    assert r["score"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# score_answer_relevance
# ---------------------------------------------------------------------------

def test_answer_relevance_on_target():
    with patch("httpx.post", return_value=_groq_ok(0.88, "Directly answers the query.")):
        r = score_answer_relevance(
            answer="90 days.", query="What is the notice period?", api_key="fake-key"
        )
    assert r["score"] == pytest.approx(0.88)


def test_answer_relevance_empty_query():
    r = score_answer_relevance("answer", "", api_key="fake-key")
    assert r["score"] is None
    assert "no query" in r["reason"]


def test_answer_relevance_empty_answer():
    r = score_answer_relevance("", "some query?", api_key="fake-key")
    assert r["score"] is None


# ---------------------------------------------------------------------------
# evaluate_trace
# ---------------------------------------------------------------------------

_TRACE = {
    "trace": {"trace_id": "t1", "query": "What is X?", "project": "p"},
    "events": [
        {"stage": "context_assembly", "attributes": {"chunks": [{"text": "X is Y."}]}},
        {"stage": "generation", "attributes": {"response": "X is Y.", "model": "gpt-4o"}},
    ],
}


def test_evaluate_trace_full():
    with patch("httpx.post", return_value=_groq_ok(0.9, "good")):
        result = evaluate_trace(_TRACE, api_key="fake-key")
    assert "faithfulness" in result
    assert "answer_relevance" in result
    assert result["faithfulness"]["score"] == pytest.approx(0.9)
    assert result["answer_relevance"]["score"] == pytest.approx(0.9)


def test_evaluate_trace_no_generation():
    trace = {
        "trace": {"trace_id": "t2", "query": "Q?", "project": "p"},
        "events": [],
    }
    # No answer → both metrics return None score
    result = evaluate_trace(trace, api_key="fake-key")
    assert result["faithfulness"]["score"] is None
    assert result["answer_relevance"]["score"] is None


def test_evaluate_trace_no_context():
    trace = {
        "trace": {"trace_id": "t3", "query": "Q?", "project": "p"},
        "events": [
            {"stage": "generation", "attributes": {"response": "Answer.", "model": "gpt-4o"}},
        ],
    }
    # No context_assembly → faithfulness None; relevance scored
    with patch("httpx.post", return_value=_groq_ok(0.7, "ok")):
        result = evaluate_trace(trace, api_key="fake-key")
    assert result["faithfulness"]["score"] is None
    assert result["answer_relevance"]["score"] == pytest.approx(0.7)
