import pytest

from ragobserve.server import metrics as M


def test_precision_recall():
    ranked = ["a", "b", "c", "d"]
    relevant = ["a", "c", "x"]
    assert M.precision_at_k(ranked, relevant, k=4) == pytest.approx(0.5)
    assert M.recall_at_k(ranked, relevant, k=4) == pytest.approx(2 / 3)
    assert M.precision_at_k([], relevant) is None
    assert M.recall_at_k(ranked, []) is None


def test_mrr():
    assert M.mrr(["x", "a", "b"], ["a"]) == pytest.approx(0.5)
    assert M.mrr(["a"], ["a"]) == 1.0
    assert M.mrr(["x", "y"], ["a"]) == 0.0


def test_ndcg_perfect_and_worst():
    assert M.ndcg(["a", "b", "c"], ["a", "b"], k=3) == pytest.approx(1.0)
    worse = M.ndcg(["c", "b", "a"], ["a"], k=3)
    assert worse is not None and worse < 1.0


def test_kendall_tau():
    assert M.kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)
    assert M.kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == pytest.approx(-1.0)
    assert M.kendall_tau(["a"], ["a"]) is None


def test_chunk_utilization():
    assert M.chunk_utilization(["a", "b", "c", "d"], ["a", "b"]) == pytest.approx(0.5)
    assert M.chunk_utilization([], ["a"]) is None


def test_evaluate_traces_aggregate():
    pairs = [
        {"trace_id": "t1", "ranked": ["a", "b"], "relevant": ["a"]},
        {"trace_id": "t2", "ranked": ["x", "a"], "relevant": ["a"]},
    ]
    out = M.evaluate_traces(pairs, k=2)
    assert out["aggregate"]["traces_evaluated"] == 2
    assert out["aggregate"]["mrr"] == pytest.approx((1.0 + 0.5) / 2)
