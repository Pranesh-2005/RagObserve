"""Built-in retrieval evaluation metrics. Pure Python, no heavy deps.

Ground-truth metrics (need ``ragobserve.log_ground_truth``): Precision@k,
Recall@k, MRR, nDCG. Derived metrics (no ground truth): chunk utilization,
reranker effectiveness (Kendall's tau between before/after orderings).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


def precision_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int = 10) -> Optional[float]:
    top = list(ranked)[:k]
    if not top:
        return None
    rel = set(relevant)
    return sum(1 for c in top if c in rel) / len(top)


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int = 10) -> Optional[float]:
    rel = set(relevant)
    if not rel:
        return None
    top = list(ranked)[:k]
    return sum(1 for c in top if c in rel) / len(rel)


def mrr(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    for i, c in enumerate(ranked):
        if c in rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg(ranked: Sequence[str], relevant: Sequence[str], k: int = 10) -> Optional[float]:
    rel = set(relevant)
    if not rel:
        return None
    top = list(ranked)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, c in enumerate(top) if c in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal > 0 else None


def kendall_tau(before: Sequence[str], after: Sequence[str]) -> Optional[float]:
    """Rank correlation over the items common to both orderings.

    1.0 = reranker changed nothing, -1.0 = fully reversed. Low absolute
    correlation means the reranker is doing significant work.
    """
    common = [c for c in before if c in set(after)]
    if len(common) < 2:
        return None
    pos = {c: i for i, c in enumerate(after)}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            d = pos[common[i]] - pos[common[j]]
            if d < 0:
                concordant += 1
            elif d > 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def chunk_utilization(retrieved_ids: Sequence[str], context_ids: Sequence[str]) -> Optional[float]:
    """Fraction of retrieved chunks that actually made it into the final context."""
    if not retrieved_ids:
        return None
    ctx = set(context_ids)
    return sum(1 for c in retrieved_ids if c in ctx) / len(retrieved_ids)


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def evaluate_traces(pairs: List[Dict[str, Any]], k: int = 10) -> Dict[str, Any]:
    """Compute per-trace and aggregate metrics for (ranked, relevant) pairs
    as produced by ``Store.traces_with_ground_truth``."""
    per_trace = []
    for p in pairs:
        ranked, relevant = p["ranked"], p["relevant"]
        per_trace.append(
            {
                "trace_id": p["trace_id"],
                "precision_at_k": precision_at_k(ranked, relevant, k),
                "recall_at_k": recall_at_k(ranked, relevant, k),
                "mrr": mrr(ranked, relevant),
                "ndcg": ndcg(ranked, relevant, k),
            }
        )
    aggregate = {
        "k": k,
        "traces_evaluated": len(per_trace),
        "precision_at_k": _mean([t["precision_at_k"] for t in per_trace]),
        "recall_at_k": _mean([t["recall_at_k"] for t in per_trace]),
        "mrr": _mean([t["mrr"] for t in per_trace]),
        "ndcg": _mean([t["ndcg"] for t in per_trace]),
    }
    return {"per_trace": per_trace, "aggregate": aggregate}


# ------------------------------------------------------------------ triage

def ids_of(results: Optional[List[Any]]) -> List[str]:
    """Chunk ids of a result list; content hash when a result has no id."""
    from ..events import content_hash

    out = []
    for item in results or []:
        if isinstance(item, dict):
            cid = item.get("id") or (content_hash(item["text"]) if item.get("text") else None)
            if cid:
                out.append(cid)
    return out


# verdict -> the stage to blame, in pipeline order
TRIAGE_VERDICTS = {
    "error": "a span raised",
    "empty_retrieval": "retrieval returned nothing",
    "retrieval_miss": "no relevant chunk was retrieved",
    "rerank_drop": "reranker dropped every relevant chunk",
    "context_drop": "relevant chunk retrieved but left out of the prompt",
    "unsupported_claims": "answer makes claims the context doesn't support",
    "unfaithful": "answer not grounded in the context",
    "off_topic": "answer doesn't address the query",
    "ok": "no failure found",
    "unknown": "not enough signal: log ground truth or run eval",
}


def triage(trace_data: Dict[str, Any], scores: Optional[Dict[str, Optional[float]]] = None,
           threshold: float = 0.5) -> Dict[str, str]:
    """Blame the *first* stage that broke, walking the pipeline in order.

    Retrieval verdicts need ground truth; answer verdicts need eval scores
    (``faithfulness`` / ``answer_relevance``) or a grounding span.
    """
    from ..events import content_hash

    scores = scores or {}
    events = trace_data.get("events") or []
    relevant = {str(x) for x in trace_data.get("ground_truth") or []}

    def verdict(v: str) -> Dict[str, str]:
        return {"verdict": v, "reason": TRIAGE_VERDICTS[v]}

    if (trace_data.get("trace") or {}).get("status") == "error" or any(
            e.get("status") == "error" for e in events):
        return verdict("error")

    # match on id *and* text hash: real apps mix schemes in one trace (int ids
    # on retrieval, none on rerank/context), and ground truth may use either
    def keys(results):
        out = set(ids_of(results))
        out |= {content_hash(r["text"]) for r in results or [] if isinstance(r, dict) and r.get("text")}
        out |= {str(r["id"]) for r in results or [] if isinstance(r, dict) and r.get("id") is not None}
        return out

    retrieved, reranked, context, grounding = set(), None, None, None
    has_retrieval = False
    for e in events:
        a, stage = e.get("attributes") or {}, e.get("stage")
        if stage in ("retrieval", "fusion"):
            has_retrieval = True
            retrieved |= keys(a.get("results"))
        elif stage == "reranking":
            reranked = keys(a.get("after"))
        elif stage == "context_assembly":
            context = keys(a.get("chunks"))
        elif stage == "grounding":
            grounding = a.get("supported_ratio")

    # an item carrying both an id and text links the schemes: relevant id 1 ->
    # that text's hash, so id-less rerank/context items still match
    for e in events:
        a = e.get("attributes") or {}
        for lst in (a.get("results"), a.get("after"), a.get("before"), a.get("chunks")):
            for r in lst or []:
                if isinstance(r, dict) and r.get("text") and r.get("id") is not None                         and str(r["id"]) in relevant:
                    relevant.add(content_hash(r["text"]))

    if has_retrieval and not retrieved:
        return verdict("empty_retrieval")
    if relevant and has_retrieval:
        if not relevant & retrieved:
            return verdict("retrieval_miss")
        if reranked is not None and not relevant & reranked:
            return verdict("rerank_drop")
        if context is not None and not relevant & context:
            return verdict("context_drop")
    if grounding is not None and grounding < threshold:
        return verdict("unsupported_claims")
    faith, rel = scores.get("faithfulness"), scores.get("answer_relevance")
    if faith is not None and faith < threshold:
        return verdict("unfaithful")
    if rel is not None and rel < threshold:
        return verdict("off_topic")
    if relevant or grounding is not None or faith is not None or rel is not None:
        return verdict("ok")
    return verdict("unknown")


def percentile(values: List[float], p: float) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    i = min(len(vals) - 1, max(0, math.ceil(p / 100 * len(vals)) - 1))
    return vals[i]
