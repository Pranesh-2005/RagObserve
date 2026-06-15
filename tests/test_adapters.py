"""Adapter tests exercise the pure translation helpers (no framework installs
needed); full integration runs are covered when the extras are installed."""
import ragobserve
from ragobserve.adapters import langchain as lc
from ragobserve.adapters import llamaindex as li


class FakeDoc:
    def __init__(self, text, source):
        self.page_content = text
        self.metadata = {"source": source}


def test_langchain_retrieval_translation(local_client):
    ev = lc.retrieval_event(
        query="notice period?",
        documents=[FakeDoc("ninety days", "msa.md"), FakeDoc("other", "nda.md")],
        trace_id="t-lc", parent_span_id=None, project="test-project",
        start_time=0.0, retriever="faiss",
    )
    assert ev["stage"] == "retrieval"
    assert ev["attributes"]["results"][0] == {
        "text": "ninety days", "metadata": {"source": "msa.md"}, "rank": 1, "source": "msa.md",
    }
    local_client.log_event(ev)
    data = local_client.store.get_trace("t-lc")
    assert data["trace"]["query"] == "notice period?"


def test_langchain_generation_translation():
    ev = lc.generation_event(
        prompts=["the prompt"], response_text="the answer", model="gpt-4o",
        trace_id="t-lc", parent_span_id=None, project="p",
        start_time=0.0, token_usage={"prompt_tokens": 10, "completion_tokens": 5},
    )
    assert ev["stage"] == "generation"
    a = ev["attributes"]
    assert (a["model"], a["input_tokens"], a["output_tokens"]) == ("gpt-4o", 10, 5)


class FakeNode:
    def __init__(self, text, score):
        self.node = type("Inner", (), {"get_content": lambda s: text, "metadata": {"file_name": "f.md"}})()
        self.score = score


def test_llamaindex_node_translation():
    results = li.nodes_to_results([FakeNode("alpha", 0.9), FakeNode("beta", 0.7)])
    assert results[0]["text"] == "alpha"
    assert results[0]["score"] == 0.9
    assert results[0]["rank"] == 1
    assert results[0]["source"] == "f.md"

    ev = li.retrieval_event("q", [FakeNode("alpha", 0.9)], "t-li", "p", 0.0)
    assert ev["stage"] == "retrieval"
    assert ev["attributes"]["results"][0]["text"] == "alpha"


def test_llamaindex_chunking_from_texts():
    # LlamaIndex exposes chunk text only via EmbeddingEndEvent.chunks (strings)
    ev = li.chunking_event(["chunk one", "chunk two"], "t-li", "p", 0.0)
    assert ev["stage"] == "chunking"
    texts = [c["text"] for c in ev["attributes"]["chunks"]]
    assert texts == ["chunk one", "chunk two"]
    assert ev["attributes"]["chunks"][0]["rank"] == 1


def test_llamaindex_generation_tokens():
    ev = li.generation_event(
        prompt="p", response_text="a", model="llama-3.1-8b-instant",
        trace_id="t-li", project="p", start_time=0.0,
        input_tokens=12, output_tokens=7,
    )
    a = ev["attributes"]
    assert (a["model"], a["input_tokens"], a["output_tokens"]) == ("llama-3.1-8b-instant", 12, 7)


def test_llamaindex_extract_usage():
    resp = type("R", (), {"raw": {"usage": {"prompt_tokens": 9, "completion_tokens": 4}}})()
    assert li._extract_usage(resp) == (9, 4)


def test_llamaindex_context_from_synthesis():
    # GetResponseStartEvent.text_chunks -> context_assembly
    ev = li.context_event("what is hybrid search", ["chunk a", "chunk b"], "t-li", "p", 0.0)
    a = ev["attributes"]
    assert ev["stage"] == "context_assembly"
    assert a["query"] == "what is hybrid search"
    assert a["final_prompt"] == "chunk a\n\nchunk b"
    assert [c["text"] for c in a["chunks"]] == ["chunk a", "chunk b"]
