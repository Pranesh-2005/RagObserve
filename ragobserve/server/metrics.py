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
