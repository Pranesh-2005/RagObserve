"""LangChain adapter: a callback handler that converts LangChain run events
into the universal RAGObserve event model.

Usage::

    from ragobserve.adapters.langchain import RagObserveCallbackHandler
    chain.invoke(question, config={"callbacks": [RagObserveCallbackHandler()]})

Requires ``pip install ragobserve[langchain]``. The translation helpers below
are pure (dict in -> event dict out) so they are testable without LangChain.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .. import client as _client
from .._diag import require_methods
from ..events import RagEvent, Stage, estimate_tokens, normalize_result
from .vectordb import _Proxy

try:
    from langchain_core.callbacks import BaseCallbackHandler  # type: ignore
except ImportError:  # pragma: no cover - exercised only without the extra
    BaseCallbackHandler = object

try:
    from langchain_core.documents.compressor import BaseDocumentCompressor as _BaseCompressor  # type: ignore
except ImportError:  # pragma: no cover
    _BaseCompressor = None


# --------------------------------------------------------------- pure mapping

def retrieval_event(query: str, documents: List[Any], trace_id: str,
                    parent_span_id: Optional[str], project: str,
                    start_time: float, retriever: Optional[str] = None) -> Dict[str, Any]:
    results = [normalize_result(d) for d in documents]
    for i, r in enumerate(results):
        r.setdefault("rank", i + 1)
        meta = r.get("metadata") or {}
        r.setdefault("source", meta.get("source"))
    ev = RagEvent(
        trace_id=trace_id, parent_span_id=parent_span_id, project=project,
        stage=Stage.RETRIEVAL.value, name=retriever or "langchain.retriever",
        start_time=start_time,
        attributes={"query": query, "results": results, "top_k": len(results), "retriever": retriever},
    )
    return ev.finish().model_dump()


def generation_event(prompts: List[str], response_text: str, model: Optional[str],
                     trace_id: str, parent_span_id: Optional[str], project: str,
                     start_time: float, token_usage: Optional[Dict[str, Any]] = None,
                     status: str = "ok") -> Dict[str, Any]:
    usage = token_usage or {}
    ev = RagEvent(
        trace_id=trace_id, parent_span_id=parent_span_id, project=project,
        stage=Stage.GENERATION.value, name=model or "langchain.llm",
        start_time=start_time,
        attributes={
            "model": model,
            "prompt": "\n\n".join(prompts),
            "response": response_text,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        },
    )
    return ev.finish(status).model_dump()


# ------------------------------------------------------------------- handler

class RagObserveCallbackHandler(BaseCallbackHandler):
    """Maps on_chain_* to trace boundaries, on_retriever_* to retrieval
    events and on_llm_* to generation events."""

    def __init__(self, project: Optional[str] = None):
        if BaseCallbackHandler is object:
            raise ImportError(
                "LangChain is not installed. Run: pip install ragobserve[langchain]"
            )
        self.project = project or _client.get_project()
        self._trace_id: Optional[str] = None
        self._root_run: Optional[str] = None
        self._starts: Dict[str, float] = {}
        self._queries: Dict[str, str] = {}
        self._trace_start: Optional[float] = None
        self._query: Optional[str] = None

    # -- chain = trace boundary -------------------------------------------
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kw):
        if parent_run_id is None and self._root_run is None:
            self._root_run = str(run_id)
            self._trace_id = RagEvent().trace_id
            self._trace_start = time.time()
            if isinstance(inputs, dict):
                for key in ("question", "query", "input"):
                    if isinstance(inputs.get(key), str):
                        self._query = inputs[key]
                        break
            elif isinstance(inputs, str):
                self._query = inputs

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kw):
        if str(run_id) == self._root_run:
            ev = RagEvent(
                trace_id=self._trace_id, project=self.project,
                stage=Stage.OTHER.value, name="langchain.chain",
                start_time=self._trace_start or time.time(),
                attributes={"query": self._query},
            )
            _client.get_client().log_event(ev.finish().model_dump())
            self._root_run = None

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kw):
        if str(run_id) == self._root_run:
            ev = RagEvent(
                trace_id=self._trace_id, project=self.project,
                stage=Stage.OTHER.value, name="langchain.chain",
                start_time=self._trace_start or time.time(),
                attributes={"query": self._query, "error": repr(error)},
            )
            _client.get_client().log_event(ev.finish("error").model_dump())
            self._root_run = None

    def _ensure_trace(self) -> str:
        if self._trace_id is None:
            self._trace_id = RagEvent().trace_id
        return self._trace_id

    # -- retriever ----------------------------------------------------------
    def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        self._starts[rid] = time.time()
        self._queries[rid] = query

    def on_retriever_end(self, documents, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        ev = retrieval_event(
            query=self._queries.pop(rid, ""),
            documents=list(documents or []),
            trace_id=self._ensure_trace(),
            parent_span_id=None,
            project=self.project,
            start_time=self._starts.pop(rid, time.time()),
        )
        _client.get_client().log_event(ev)

    # -- llm ------------------------------------------------------------------
    def _log_context(self, final_prompt: str) -> None:
        """The prompt sent to the model is exactly the assembled context, so
        emit a context_assembly event — no manual ``log_context`` needed."""
        ev = RagEvent(
            trace_id=self._ensure_trace(), project=self.project,
            stage=Stage.CONTEXT_ASSEMBLY.value, name="langchain.prompt",
            attributes={"final_prompt": final_prompt, "query": None,
                        "system_prompt": None, "chunks": [],
                        "token_count": estimate_tokens(final_prompt),
                        "context_window": None},
        )
        _client.get_client().log_event(ev.finish().model_dump())

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        self._starts[rid] = time.time()
        self._queries[rid] = "\n\n".join(prompts)
        self._log_context(self._queries[rid])

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        self._starts[rid] = time.time()
        flat = []
        for batch in messages:
            for m in batch:
                flat.append(f"{getattr(m, 'type', 'msg')}: {getattr(m, 'content', m)}")
        self._queries[rid] = "\n".join(flat)
        self._log_context(self._queries[rid])

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        text = ""
        model = None
        usage = None
        try:
            out = getattr(response, "llm_output", None) or {}
            model = out.get("model_name") or out.get("model")
            usage = out.get("token_usage") or out.get("usage")
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            text = getattr(gen, "text", "") or getattr(msg, "content", "")
            # chat models put usage on the message, not llm_output
            meta = getattr(msg, "response_metadata", None) or {}
            model = model or meta.get("model_name") or meta.get("model")
            usage = usage or meta.get("token_usage")
            um = getattr(msg, "usage_metadata", None)
            if not usage and um:
                usage = {"prompt_tokens": um.get("input_tokens"),
                         "completion_tokens": um.get("output_tokens")}
        except (AttributeError, IndexError):
            pass
        ev = generation_event(
            prompts=[self._queries.pop(rid, "")], response_text=text, model=model,
            trace_id=self._ensure_trace(), parent_span_id=None, project=self.project,
            start_time=self._starts.pop(rid, time.time()), token_usage=usage,
        )
        _client.get_client().log_event(ev)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kw):
        rid = str(run_id)
        ev = generation_event(
            prompts=[self._queries.pop(rid, "")], response_text=repr(error), model=None,
            trace_id=self._ensure_trace(), parent_span_id=None, project=self.project,
            start_time=self._starts.pop(rid, time.time()), status="error",
        )
        _client.get_client().log_event(ev)


# ----------------------------------------------------- ingest-time instrumenting
# LangChain text splitters and embeddings emit no callbacks (they are plain
# batch calls), so the callback handler above can never see them. These thin
# proxies wrap the objects and log a chunking / embedding event per call.

class _SplitterProxy(_Proxy):
    def _emit(self, docs):
        from ..tracing import log_chunks
        try:
            log_chunks(
                list(docs or []),
                strategy=type(self._target).__name__,
                chunk_size=getattr(self._target, "_chunk_size", None),
                overlap=getattr(self._target, "_chunk_overlap", None),
            )
        except Exception:
            pass
        return docs

    def split_documents(self, *a, **k):
        return self._emit(self._target.split_documents(*a, **k))

    def split_text(self, *a, **k):
        return self._emit(self._target.split_text(*a, **k))

    def create_documents(self, *a, **k):
        return self._emit(self._target.create_documents(*a, **k))

    def transform_documents(self, *a, **k):
        return self._emit(self._target.transform_documents(*a, **k))


def instrument_splitter(splitter: Any) -> Any:
    """Wrap a LangChain ``TextSplitter`` so ``split_documents`` / ``split_text``
    / ``create_documents`` / ``transform_documents`` auto-log a chunking event."""
    require_methods(splitter, ["split_documents", "split_text", "create_documents",
                               "transform_documents"], "instrument_splitter")
    return _SplitterProxy(splitter, "langchain")


class _LoaderProxy(_Proxy):
    def _emit(self, docs):
        from ..tracing import log_ingestion

        docs = list(docs or [])
        try:
            srcs = [(getattr(d, "metadata", {}) or {}).get("source") for d in docs]
            log_ingestion(count=len(docs), sources=[s for s in srcs if s][:50])
        except Exception:
            pass
        return docs

    def load(self, *a, **k):
        return self._emit(self._target.load(*a, **k))

    def load_and_split(self, *a, **k):
        return self._emit(self._target.load_and_split(*a, **k))


def instrument_loader(loader: Any) -> Any:
    """Wrap a LangChain ``BaseLoader`` so ``load`` / ``load_and_split`` auto-log
    an ingestion event (document count + sources). ``lazy_load`` passes through
    untouched (streaming iterator)."""
    require_methods(loader, ["load", "load_and_split"], "instrument_loader")
    return _LoaderProxy(loader, "langchain")


# Rerankers are ``BaseDocumentCompressor``s and ``compress_documents`` fires no
# callback, so the handler can't see reranking. Wrap the compressor in a real
# subclass (so ``ContextualCompressionRetriever`` still validates it) that logs
# a reranking event with before/after order.
if _BaseCompressor is not None:

    class _LoggedCompressor(_BaseCompressor):  # type: ignore[misc, valid-type]
        target: Any = None
        model_config = {"arbitrary_types_allowed": True}

        def compress_documents(self, documents, query, callbacks=None):
            from ..tracing import log_rerank

            before = list(documents or [])
            after = list(self.target.compress_documents(before, query, callbacks))
            try:
                inner = getattr(self.target, "model", None)
                model = (getattr(inner, "model_name", None)
                         or getattr(inner, "model", None)
                         or type(inner if inner is not None else self.target).__name__)
                log_rerank(before, after, model=model, top_n=getattr(self.target, "top_n", None))
            except Exception:
                pass
            return after


def instrument_compressor(compressor: Any) -> Any:
    """Wrap a LangChain reranker / ``BaseDocumentCompressor`` so
    ``compress_documents`` auto-logs a reranking event (before/after, model,
    top_n). Pass the wrapped compressor to ``ContextualCompressionRetriever``."""
    if _BaseCompressor is None:
        raise ImportError("LangChain is not installed. Run: pip install ragobserve[langchain]")
    require_methods(compressor, ["compress_documents"], "instrument_compressor")
    return _LoggedCompressor(target=compressor)


def instrument_embeddings(embeddings: Any) -> Any:
    """Wrap a LangChain ``Embeddings`` so ``embed_documents`` auto-logs an
    embedding event. ``embed_query`` passes straight through (query embeds are
    already implied by the retrieval event).

    Returns a real ``Embeddings`` subclass — not a generic proxy — so callers
    that ``isinstance``-check the object (e.g. FAISS / vector stores) keep
    working. Any other attribute/method delegates to the wrapped object.
    """
    require_methods(embeddings, ["embed_documents"], "instrument_embeddings")
    try:
        from langchain_core.embeddings import Embeddings as _Base
    except Exception:  # langchain not installed -> degrade to a plain proxy
        _Base = object

    target = embeddings

    class _LoggedEmbeddings(_Base):  # type: ignore[misc, valid-type]
        def embed_documents(self, texts, *a, **k):
            from ..tracing import log_embedding

            t0 = time.time()
            vecs = target.embed_documents(texts, *a, **k)
            dur = (time.time() - t0) * 1000.0
            try:
                log_embedding(
                    model=(getattr(target, "model", None)
                           or getattr(target, "model_name", None)
                           or type(target).__name__),
                    input_count=len(texts) if texts is not None else (len(vecs) if vecs else 0),
                    dimensions=len(vecs[0]) if vecs else None,
                    duration_ms=dur,
                )
            except Exception:
                pass
            return vecs

        def embed_query(self, text, *a, **k):
            return target.embed_query(text, *a, **k)

        def __getattr__(self, name):
            return getattr(target, name)

    return _LoggedEmbeddings()
