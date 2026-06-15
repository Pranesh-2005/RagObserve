"""End-to-end adapter behaviour: diagnostics, the vector-DB wrappers, and the
LangChain / LlamaIndex dispatch paths (not just the pure mapping helpers)."""
import json
import warnings
from uuid import uuid4

import pytest

from ragobserve._diag import RagObserveWarning, require_methods


def _events(client, stage, project="test-project"):
    rows = client.store._conn.execute(
        "select attributes from events where project=? and stage=?", (project, stage)
    ).fetchall()
    return [json.loads(a) for (a,) in rows]


# ----------------------------------------------------------------- diagnostics

def test_require_methods_warns_when_all_missing():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        require_methods(object(), ["nope", "nada"], "x")
    assert any(issubclass(x.category, RagObserveWarning) for x in w)


def test_require_methods_silent_when_one_present():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        require_methods({}, ["nope", "keys"], "x")  # dict has .keys
    assert not w


# ------------------------------------------------------------ vector-DB wrappers

def test_chroma_wrapper_logs_and_passes_through(local_client):
    from ragobserve.adapters.vectordb import instrument_chroma

    class Col:
        def query(self, query_texts=None, n_results=3):
            return {"ids": [["a"]], "documents": [["doc"]],
                    "distances": [[0.1]], "metadatas": [[{"source": "s"}]]}

        def add(self, **k):
            return "passthrough"

    c = instrument_chroma(Col())
    assert c.add(x=1) == "passthrough"          # delegation
    c.query(query_texts=["q"], n_results=1)
    ev = _events(local_client, "retrieval")[0]
    assert ev["retriever"] == "chroma"
    assert ev["results"][0]["text"] == "doc"


def test_qdrant_and_pgvector_wrappers(local_client):
    from ragobserve.adapters.vectordb import instrument_qdrant, log_pgvector

    class SP:
        def __init__(self):
            self.id = "q1"; self.score = 0.7; self.payload = {"page_content": "qd"}

    class QC:
        def search(self, **k):
            return [SP()]
        def query_points(self, **k):
            return [SP()]

    instrument_qdrant(QC()).search(collection_name="d", query_vector=[0.1], limit=5)
    log_pgvector("how many days", [{"text": "pg", "distance": 0.05, "source": "d"}], top_k=5)

    retrievers = {e["retriever"] for e in _events(local_client, "retrieval")}
    assert {"qdrant", "pgvector"} <= retrievers


# ----------------------------------------------------- LangChain callback handler

def test_langchain_handler_full_dispatch(local_client):
    pytest.importorskip("langchain_core")
    from ragobserve.adapters.langchain import RagObserveCallbackHandler

    h = RagObserveCallbackHandler(project="test-project")

    class Doc:
        def __init__(self, t):
            self.page_content = t; self.metadata = {"source": "f"}

    rid = uuid4()
    h.on_retriever_start({}, "q", run_id=rid)
    h.on_retriever_end([Doc("hit")], run_id=rid)

    rid2 = uuid4()
    msg_in = type("Hum", (), {"type": "human", "content": "CONTEXT then question"})()
    h.on_chat_model_start({}, [[msg_in]], run_id=rid2)

    ai = type("AIMsg", (), {"content": "answer",
                            "response_metadata": {"model_name": "gpt-x"},
                            "usage_metadata": {"input_tokens": 10, "output_tokens": 5}})()
    gen = type("Gen", (), {"text": "", "message": ai})()
    resp = type("LLMResult", (), {"generations": [[gen]], "llm_output": {}})()
    h.on_llm_end(resp, run_id=rid2)

    retr = _events(local_client, "retrieval")
    ctx = _events(local_client, "context_assembly")
    gen_e = _events(local_client, "generation")
    assert retr and retr[0]["results"][0]["text"] == "hit"
    assert ctx and "CONTEXT then question" in ctx[0]["final_prompt"]   # auto, no manual log_context
    a = gen_e[0]
    assert a["model"] == "gpt-x" and (a["input_tokens"], a["output_tokens"]) == (10, 5)


def test_instrument_embeddings_is_real_subclass(local_client):
    pytest.importorskip("langchain_core")
    from langchain_core.embeddings import Embeddings

    from ragobserve.adapters import instrument_embeddings

    class Fake(Embeddings):
        def embed_documents(self, texts):
            return [[0.0] * 4 for _ in texts]
        def embed_query(self, text):
            return [0.0] * 4

    w = instrument_embeddings(Fake())
    assert isinstance(w, Embeddings)               # FAISS-safe
    w.embed_documents(["a", "b"])
    assert _events(local_client, "embedding")[0]["input_count"] == 2


def test_instrument_compressor_logs_rerank(local_client):
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from langchain_core.documents.compressor import BaseDocumentCompressor

    from ragobserve.adapters import instrument_compressor

    class RR(BaseDocumentCompressor):
        model_config = {"arbitrary_types_allowed": True}
        def compress_documents(self, documents, query, callbacks=None):
            return list(documents)[:1]

    w = instrument_compressor(RR())
    assert isinstance(w, BaseDocumentCompressor)    # CCR-safe
    w.compress_documents([Document(page_content="a"), Document(page_content="b")], "q")
    a = _events(local_client, "reranking")[0]
    assert len(a["before"]) == 2 and len(a["after"]) == 1


# ------------------------------------------------------ LlamaIndex event handler

def test_llamaindex_handler_stages(local_client):
    from ragobserve.adapters import llamaindex as LI

    h = LI.RagObserveEventHandler.__new__(LI.RagObserveEventHandler)
    object.__setattr__(h, "_state", {"trace_id": None, "starts": {}, "query": None})
    LI.RagObserveEventHandler.project_name = "test-project"

    def mk(name, **fields):
        return type(name, (), fields)()

    class NWS:
        def __init__(self, t):
            self.node = type("N", (), {"get_content": lambda self: t, "metadata": {}})()
            self.score = 0.5

    h._handle(mk("EmbeddingStartEvent", model_dict={"model_name": "bge"}))
    h._handle(mk("EmbeddingEndEvent", chunks=["c1", "c2"], embeddings=[[0.0] * 8]))
    h._handle(mk("RetrievalStartEvent"))
    h._handle(mk("RetrievalEndEvent", str_or_query_bundle="q", nodes=[NWS("r")]))
    h._handle(mk("LLMCompletionStartEvent", model_dict={"model": "llama"}))
    resp = type("R", (), {"text": "ans", "raw": {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}})()
    h._handle(mk("LLMCompletionEndEvent", prompt="p", response=resp))

    for stage in ("embedding", "chunking", "retrieval", "generation"):
        assert _events(local_client, stage), f"missing {stage}"
    g = _events(local_client, "generation")[0]
    assert g["model"] == "llama" and (g["input_tokens"], g["output_tokens"]) == (7, 3)


def test_instrument_postprocessor_logs_rerank(local_client):
    pytest.importorskip("llama_index.core")
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, TextNode

    from ragobserve.adapters.llamaindex import instrument_postprocessor

    class RR(BaseNodePostprocessor):
        @classmethod
        def class_name(cls):
            return "RR"
        def _postprocess_nodes(self, nodes, query_bundle=None):
            return list(nodes)[:1]

    w = instrument_postprocessor(RR())
    assert isinstance(w, BaseNodePostprocessor)
    w.postprocess_nodes(
        [NodeWithScore(node=TextNode(text="a")), NodeWithScore(node=TextNode(text="b"))],
        query_str="q",
    )
    a = _events(local_client, "reranking")[0]
    assert len(a["before"]) == 2 and len(a["after"]) == 1
