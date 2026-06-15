import ragobserve
from ragobserve.tracing import current_trace_id


def get_trace(client, trace_id):
    return client.store.get_trace(trace_id)


def test_trace_context_manager_nesting(local_client):
    with ragobserve.trace("query", query="what is x?") as root:
        tid = current_trace_id()
        assert tid == root.trace_id
        with ragobserve.trace("retrieval") as child:
            assert child.trace_id == tid
            assert child.parent_span_id == root.span_id

    data = get_trace(local_client, tid)
    assert data is not None
    assert data["trace"]["query"] == "what is x?"
    stages = [e["stage"] for e in data["events"]]
    assert "retrieval" in stages
    assert len(data["events"]) == 2


def test_trace_decorator(local_client):
    captured = {}

    @ragobserve.trace
    def retrieve():
        captured["tid"] = current_trace_id()
        return 42

    assert retrieve() == 42
    data = get_trace(local_client, captured["tid"])
    assert data["events"][0]["stage"] == "retrieval"  # inferred from function name


def test_trace_error_status(local_client):
    tid = None
    try:
        with ragobserve.trace("query", query="boom") as span:
            tid = span.trace_id
            raise ValueError("kaput")
    except ValueError:
        pass
    data = get_trace(local_client, tid)
    assert data["trace"]["status"] == "error"
    assert data["events"][0]["status"] == "error"
    assert "kaput" in data["events"][0]["attributes"]["error"]


def test_convenience_loggers_shape(local_client):
    results = [{"text": "alpha", "score": 0.9}, {"text": "beta", "score": 0.7}]
    with ragobserve.trace("query", query="q") as span:
        tid = span.trace_id
        ragobserve.log_retrieval("q", results, retriever="faiss")
        ragobserve.log_rerank(results, list(reversed(results)), model="bge")
        ragobserve.log_context("PROMPT", system_prompt="SYS", query="q", chunks=results)
        ragobserve.log_generation(model="gpt-4o", response="ans", cost=0.001)

    data = get_trace(local_client, tid)
    by_stage = {e["stage"]: e for e in data["events"]}
    assert by_stage["retrieval"]["attributes"]["results"][0]["rank"] == 1
    assert by_stage["reranking"]["attributes"]["after"][0]["text"] == "beta"
    assert by_stage["context_assembly"]["attributes"]["final_prompt"] == "PROMPT"
    assert by_stage["generation"]["attributes"]["cost"] == 0.001
    # chunk analytics populated from retrieval results
    chunks = local_client.store.chunk_views("test-project", "top")
    assert len(chunks) == 2


def test_ground_truth_and_metrics(local_client):
    results = [{"text": "relevant chunk", "score": 0.9}, {"text": "noise", "score": 0.5}]
    with ragobserve.trace("query", query="q"):
        ragobserve.log_retrieval("q", results)
        ragobserve.log_ground_truth([ragobserve.Chunk(text="relevant chunk").hashed()])

    pairs = local_client.store.traces_with_ground_truth("test-project")
    assert len(pairs) == 1
    assert pairs[0]["ranked"][0] == ragobserve.Chunk(text="relevant chunk").hashed()
