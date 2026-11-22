"""LLM-as-judge evaluation metrics for RAG systems.

Faithfulness   — does the answer stay grounded in the retrieved context?
Answer relevance — does the answer address the original query?

Judges through the generation-replay provider registry (``server/llm.py``), so
any provider works; Groq is the default. Set its key env var (``GROQ_API_KEY``)
or pass ``api_key`` explicitly. For Ragas metrics instead, see ``ragobserve.ragas``.

Usage::

    from ragobserve.eval import score_faithfulness, score_answer_relevance

    faith = score_faithfulness(answer="90 days.", context=["Notice period is 90 days."])
    rel   = score_answer_relevance(answer="90 days.", query="What is the notice period?")
    # {"score": 0.97, "reason": "..."}
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_FAITHFULNESS_PROMPT = """\
You are evaluating a RAG (retrieval-augmented generation) system.
Given retrieved context chunks and a generated answer, score how faithfully \
the answer is grounded in the context.

CONTEXT:
{context}

ANSWER:
{answer}

Respond ONLY with a single JSON object on one line:
{{"score": <0.0-1.0>, "reason": "<one concise sentence>"}}

Score 1.0 = every claim in the answer is directly supported by the context.
Score 0.0 = the answer contradicts or completely ignores the context."""

_RELEVANCE_PROMPT = """\
You are evaluating a RAG (retrieval-augmented generation) system.
Given a user query and a generated answer, score how well the answer addresses the query.

QUERY:
{query}

ANSWER:
{answer}

Respond ONLY with a single JSON object on one line:
{{"score": <0.0-1.0>, "reason": "<one concise sentence>"}}

Score 1.0 = the answer directly and completely answers the query.
Score 0.0 = the answer is off-topic or does not address the query at all."""

DEFAULT_PROVIDER = "groq"


def _call_llm(
    prompt: str,
    model: Optional[str] = None,
    api_key: str = "",
    provider: str = DEFAULT_PROVIDER,
) -> Dict[str, Any]:
    """Judge call through the same provider registry generation replay uses, so
    any of its providers can judge. ``model=None`` → the provider's default."""
    from .server import llm

    # 1024, not ~150: reasoning models (gpt-oss, o-series) spend the budget on
    # hidden reasoning first; 150 left Groq's gpt-oss-20b with ~30 tokens of answer.
    out = llm.generate(provider, model or "", prompt, max_tokens=1024,
                       temperature=0.0, api_key=api_key or None)
    text = (out.get("text") or "").strip()
    # Try full text first (handles nested JSON), then extract first {...} block
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"LLM returned non-JSON: {text!r}")
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        raise ValueError(f"LLM returned non-JSON: {text!r}")


def score_faithfulness(
    answer: str,
    context: List[Any],
    model: Optional[str] = None,
    api_key: str = "",
    provider: str = DEFAULT_PROVIDER,
) -> Dict[str, Any]:
    """Score 0–1: how grounded is the answer in the retrieved context?

    Args:
        answer:  The generated response to evaluate.
        context: Retrieved chunks — list of str or dict with a ``text`` key.
        model:    Judge model (default: the provider's default model).
        api_key:  Provider key (falls back to the provider's env var, e.g. ``GROQ_API_KEY``).
        provider: Any provider from ``ragobserve providers`` (default: groq).

    Returns:
        ``{"score": float, "reason": str}``
        ``score`` is ``None`` when context or answer is empty.
    """
    ctx_text = "\n---\n".join(
        (c.get("text") or "") if isinstance(c, dict) else str(c)
        for c in (context or [])
    ).strip()
    if not ctx_text:
        return {"score": None, "reason": "no context available to evaluate faithfulness"}
    if not (answer or "").strip():
        return {"score": None, "reason": "no answer to evaluate"}

    result = _call_llm(_FAITHFULNESS_PROMPT.format(context=ctx_text, answer=answer), model, api_key, provider)
    return {"score": float(result.get("score", 0)), "reason": str(result.get("reason", ""))}


def score_answer_relevance(
    answer: str,
    query: str,
    model: Optional[str] = None,
    api_key: str = "",
    provider: str = DEFAULT_PROVIDER,
) -> Dict[str, Any]:
    """Score 0–1: how well does the answer address the original query?

    Returns:
        ``{"score": float, "reason": str}``
    """
    if not (query or "").strip():
        return {"score": None, "reason": "no query to evaluate against"}
    if not (answer or "").strip():
        return {"score": None, "reason": "no answer to evaluate"}

    result = _call_llm(_RELEVANCE_PROMPT.format(query=query, answer=answer), model, api_key, provider)
    return {"score": float(result.get("score", 0)), "reason": str(result.get("reason", ""))}


def evaluate_trace(
    trace_data: Dict[str, Any],
    model: Optional[str] = None,
    api_key: str = "",
    provider: str = DEFAULT_PROVIDER,
) -> Dict[str, Any]:
    """Run all eval metrics on a fully-loaded trace dict (from ``get_trace``).

    Returns::

        {
            "faithfulness":     {"score": 0.95, "reason": "..."},
            "answer_relevance": {"score": 0.88, "reason": "..."},
        }
    """
    query = trace_data.get("trace", {}).get("query") or ""
    answer, chunks = "", []
    for ev in trace_data.get("events", []):
        a = ev.get("attributes") or {}
        if ev.get("stage") == "generation" and not answer:
            answer = a.get("response") or ""
        if ev.get("stage") == "context_assembly" and not chunks:
            chunks = a.get("chunks") or []

    return {
        "faithfulness": score_faithfulness(answer, chunks, model, api_key, provider),
        "answer_relevance": score_answer_relevance(answer, query, model, api_key, provider),
    }
