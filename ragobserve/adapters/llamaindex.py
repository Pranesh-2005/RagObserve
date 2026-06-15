"""LlamaIndex adapter: an instrumentation event handler that converts
LlamaIndex events into the universal RAGObserve event model.

Usage::

    from ragobserve.adapters.llamaindex import register
    register()  # before building your index / query engine

Requires ``pip install ragobserve[llamaindex]``. The translation helpers are
pure so they are testable without LlamaIndex installed.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .. import client as _client
from .._diag import require_methods, warn
from ..events import RagEvent, Stage, normalize_result

try:
    from llama_index.core.instrumentation import get_dispatcher  # type: ignore
    from llama_index.core.instrumentation.event_handlers import BaseEventHandler  # type: ignore

    _LLAMAINDEX = True
except ImportError:  # pragma: no cover - exercised only without the extra
    BaseEventHandler = object
    _LLAMAINDEX = False

try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor as _BasePostprocessor  # type: ignore
except ImportError:  # pragma: no cover
    _BasePostprocessor = None


# --------------------------------------------------------------- pure mapping

def nodes_to_results(nodes: List[Any]) -> List[Dict[str, Any]]:
    """Convert LlamaIndex NodeWithScore-like objects (or dicts) to wire results."""
    results = []
    for i, n in enumerate(nodes or []):
        if isinstance(n, dict):
            r = normalize_result(n)
        else:
            inner = getattr(n, "node", n)
            text = ""
            get_content = getattr(inner, "get_content", None)
            if callable(get_content):
                try:
                    text = get_content()
                except Exception:
                    text = getattr(inner, "text", "") or ""
            else:
                text = getattr(inner, "text", "") or ""
            r = {
                "text": text,
                "score": getattr(n, "score", None),
                "metadata": dict(getattr(inner, "metadata", {}) or {}),
            }
        r.setdefault("rank", i + 1)
        meta = r.get("metadata") or {}
        r.setdefault("source", meta.get("file_name") or meta.get("source"))
        results.append(r)
    return results


def retrieval_event(query: str, nodes: List[Any], trace_id: str, project: str,
                    start_time: float) -> Dict[str, Any]:
    results = nodes_to_results(nodes)
    ev = RagEvent(
        trace_id=trace_id, project=project, stage=Stage.RETRIEVAL.value,
        name="llamaindex.retriever", start_time=start_time,
        attributes={"query": query, "results": results, "top_k": len(results),
                    "retriever": "llamaindex"},
    )
    return ev.finish().model_dump()


def chunking_event(items: List[Any], trace_id: str, project: str,
                   start_time: float) -> Dict[str, Any]:
    """``items`` may be LlamaIndex nodes or plain chunk-text strings (the latter
    is what ``EmbeddingEndEvent.chunks`` carries — the only ingest event that
    exposes chunk text in current LlamaIndex)."""
    chunks = []
    for i, it in enumerate(items or []):
        c = {"text": it} if isinstance(it, str) else nodes_to_results([it])[0]
        c.setdefault("rank", i + 1)
        chunks.append(c)
    ev = RagEvent(
        trace_id=trace_id, project=project, stage=Stage.CHUNKING.value,
        name="llamaindex.node_parser", start_time=start_time,
        attributes={"chunks": chunks, "strategy": "llamaindex.node_parser",
                    "chunk_size": None, "overlap": None},
    )
    return ev.finish().model_dump()


def context_event(query: str, text_chunks: List[Any], trace_id: str, project: str,
                  start_time: float) -> Dict[str, Any]:
    """Synthesis context: ``GetResponseStartEvent.text_chunks`` is exactly the
    context handed to the LLM, so capture it as a context_assembly event."""
    from ..events import estimate_tokens

    chunks = [{"text": str(c)} for c in (text_chunks or [])]
    final = "\n\n".join(str(c) for c in (text_chunks or []))
    ev = RagEvent(
        trace_id=trace_id, project=project, stage=Stage.CONTEXT_ASSEMBLY.value,
        name="llamaindex.synthesizer", start_time=start_time,
        attributes={"query": query, "final_prompt": final, "system_prompt": None,
                    "chunks": chunks, "token_count": estimate_tokens(final),
                    "context_window": None},
    )
    return ev.finish().model_dump()


def _extract_usage(resp: Any):
    """Best-effort prompt/completion token counts from a LlamaIndex LLM
    response (raw provider ``usage`` first, then ``additional_kwargs``)."""
    raw = getattr(resp, "raw", None)
    usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
    if usage is not None:
        g = (lambda k: usage.get(k)) if isinstance(usage, dict) else (lambda k: getattr(usage, k, None))
        inp, out = g("prompt_tokens"), g("completion_tokens")
        if inp is not None or out is not None:
            return inp, out
    ak = getattr(resp, "additional_kwargs", None) or {}
    return ak.get("prompt_tokens"), ak.get("completion_tokens")


def embedding_event(model: Optional[str], input_count: int, dimensions: Optional[int],
                    trace_id: str, project: str, start_time: float) -> Dict[str, Any]:
    ev = RagEvent(
        trace_id=trace_id, project=project, stage=Stage.EMBEDDING.value,
        name=model or "llamaindex.embedding", start_time=start_time,
        attributes={"model": model, "input_count": input_count,
                    "dimensions": dimensions, "cost": None},
    )
    return ev.finish().model_dump()


def generation_event(prompt: str, response_text: str, model: Optional[str],
                     trace_id: str, project: str, start_time: float,
                     input_tokens: Optional[int] = None,
                     output_tokens: Optional[int] = None) -> Dict[str, Any]:
    ev = RagEvent(
        trace_id=trace_id, project=project, stage=Stage.GENERATION.value,
        name=model or "llamaindex.llm", start_time=start_time,
        attributes={"model": model, "prompt": prompt, "response": response_text,
                    "input_tokens": input_tokens, "output_tokens": output_tokens},
    )
    return ev.finish().model_dump()


# ------------------------------------------------------------------- handler

class RagObserveEventHandler(BaseEventHandler):  # type: ignore[misc]
    """Maps LlamaIndex instrumentation events onto RAGObserve stages."""

    project_name: str = "default"

    @classmethod
    def class_name(cls) -> str:
        return "RagObserveEventHandler"

    def __init__(self, **data):
        if not _LLAMAINDEX:
            raise ImportError(
                "LlamaIndex is not installed. Run: pip install ragobserve[llamaindex]"
            )
        data.setdefault("project_name", _client.get_project())
        super().__init__(**data)
        # mutable handler state lives outside pydantic fields
        object.__setattr__(self, "_state", {"trace_id": None, "starts": {}, "query": None})

    def _trace(self) -> str:
        st = self._state
        if st["trace_id"] is None:
            st["trace_id"] = RagEvent().trace_id
        return st["trace_id"]

    def handle(self, event, **kwargs) -> None:
        try:
            self._handle(event)
        except Exception:
            # observability must never crash the instrumented app
            pass

    def _handle(self, event) -> None:
        name = type(event).__name__
        st = self._state
        if name == "QueryStartEvent":
            st["trace_id"] = RagEvent().trace_id
            st["query"] = str(getattr(event, "query", ""))
            st["starts"]["query"] = time.time()
            st["gen_logged"] = False
        elif name == "QueryEndEvent":
            ev = RagEvent(
                trace_id=self._trace(), project=self.project_name,
                stage=Stage.OTHER.value, name="llamaindex.query",
                start_time=st["starts"].pop("query", time.time()),
                attributes={"query": st.get("query")},
            )
            _client.get_client().log_event(ev.finish().model_dump())
            st["trace_id"] = None
        # ---- chat engine boundary (.chat / .stream_chat / agent loop) --------
        elif name in ("StreamChatStartEvent", "AgentChatWithStepStartEvent"):
            st["trace_id"] = RagEvent().trace_id
            st["query"] = str(getattr(event, "user_msg", "") or st.get("query") or "")
            st["starts"]["chat"] = time.time()
            st["gen_logged"] = False
            st["stream_deltas"] = []
        elif name == "StreamChatDeltaReceivedEvent":
            st.setdefault("stream_deltas", []).append(str(getattr(event, "delta", "") or ""))
        elif name in ("StreamChatEndEvent", "AgentChatWithStepEndEvent"):
            # only synthesise a generation if the LLM events didn't already log
            # one this turn (true for the streaming path)
            if not st.get("gen_logged"):
                resp = getattr(event, "response", None)
                text = "".join(st.get("stream_deltas") or []) or (str(resp) if resp is not None else "")
                if text:
                    _client.get_client().log_event(
                        generation_event(
                            prompt=st.get("query") or "", response_text=text,
                            model=st.get("llm_model"), trace_id=self._trace(),
                            project=self.project_name,
                            start_time=st["starts"].pop("chat", time.time()),
                        )
                    )
            st["stream_deltas"] = []
            st["trace_id"] = None
        # ---- synthesis: the context actually handed to the LLM ---------------
        elif name in ("GetResponseStartEvent", "GetMessageResponseStartEvent"):
            chunks = getattr(event, "text_chunks", None) or getattr(event, "message_chunks", None) or []
            _client.get_client().log_event(
                context_event(
                    query=str(getattr(event, "query_str", "") or st.get("query") or ""),
                    text_chunks=list(chunks), trace_id=self._trace(),
                    project=self.project_name, start_time=time.time(),
                )
            )
        elif name == "RetrievalStartEvent":
            st["starts"]["retrieval"] = time.time()
        elif name == "RetrievalEndEvent":
            _client.get_client().log_event(
                retrieval_event(
                    query=str(getattr(event, "str_or_query_bundle", st.get("query") or "")),
                    nodes=list(getattr(event, "nodes", []) or []),
                    trace_id=self._trace(), project=self.project_name,
                    start_time=st["starts"].pop("retrieval", time.time()),
                )
            )
        elif name in ("EmbeddingStartEvent", "SparseEmbeddingStartEvent"):
            st["starts"]["embed"] = time.time()
            md = getattr(event, "model_dict", None) or {}
            st["embed_model"] = md.get("model_name") or md.get("model")
        elif name in ("EmbeddingEndEvent", "SparseEmbeddingEndEvent"):
            chunks = list(getattr(event, "chunks", []) or [])
            embs = list(getattr(event, "embeddings", []) or [])
            start = st["starts"].pop("embed", time.time())
            _client.get_client().log_event(
                embedding_event(
                    model=st.get("embed_model"),
                    input_count=len(chunks),
                    dimensions=len(embs[0]) if embs else None,
                    trace_id=self._trace(), project=self.project_name,
                    start_time=start,
                )
            )
            # LlamaIndex emits no node-parsing event; a batch embed (>1 chunk)
            # is the ingest pass, so its chunk texts are the chunks.
            if len(chunks) > 1:
                _client.get_client().log_event(
                    chunking_event(chunks, trace_id=self._trace(),
                                   project=self.project_name, start_time=start)
                )
        elif name == "ReRankStartEvent":
            st["starts"]["rerank"] = time.time()
            st["rerank_before"] = nodes_to_results(list(getattr(event, "nodes", []) or []))
            st["rerank_model"] = getattr(event, "model_name", None)  # only on Start
            st["rerank_top_n"] = getattr(event, "top_n", None)
        elif name == "ReRankEndEvent":
            ev = RagEvent(
                trace_id=self._trace(), project=self.project_name,
                stage=Stage.RERANKING.value, name="llamaindex.reranker",
                start_time=st["starts"].pop("rerank", time.time()),
                attributes={
                    "before": st.pop("rerank_before", []),
                    "after": nodes_to_results(list(getattr(event, "nodes", []) or [])),
                    "model": st.pop("rerank_model", None),
                    "top_n": st.pop("rerank_top_n", None),
                },
            )
            _client.get_client().log_event(ev.finish().model_dump())
        elif name in ("LLMCompletionStartEvent", "LLMChatStartEvent"):
            st["starts"]["llm"] = time.time()
            md = getattr(event, "model_dict", None) or {}
            st["llm_model"] = md.get("model") or md.get("model_name")
        elif name == "LLMCompletionEndEvent":
            resp = getattr(event, "response", None)
            inp, out = _extract_usage(resp)
            _client.get_client().log_event(
                generation_event(
                    prompt=str(getattr(event, "prompt", "")),
                    response_text=str(getattr(resp, "text", "")),
                    model=st.get("llm_model"), trace_id=self._trace(),
                    project=self.project_name,
                    start_time=st["starts"].pop("llm", time.time()),
                    input_tokens=inp, output_tokens=out,
                )
            )
            st["gen_logged"] = True
        elif name == "LLMChatEndEvent":
            resp = getattr(event, "response", None)
            msg = getattr(resp, "message", None)
            inp, out = _extract_usage(resp)
            _client.get_client().log_event(
                generation_event(
                    prompt="\n".join(str(m) for m in (getattr(event, "messages", []) or [])),
                    response_text=str(getattr(msg, "content", "") or ""),
                    model=st.get("llm_model"), trace_id=self._trace(),
                    project=self.project_name,
                    start_time=st["starts"].pop("llm", time.time()),
                    input_tokens=inp, output_tokens=out,
                )
            )
            st["gen_logged"] = True


def register(project: Optional[str] = None) -> "RagObserveEventHandler":
    """Attach the RAGObserve handler to LlamaIndex's root dispatcher."""
    if not _LLAMAINDEX:
        raise ImportError("LlamaIndex is not installed. Run: pip install ragobserve[llamaindex]")
    global _REGISTERED_HANDLER
    _check_event_drift()
    handler = RagObserveEventHandler(project_name=project or _client.get_project())
    get_dispatcher().add_event_handler(handler)
    _REGISTERED_HANDLER = handler
    return handler


# event class names the handler dispatches on; if a LlamaIndex version renames
# or drops one, the corresponding stage silently stops being captured — warn.
_EXPECTED_EVENTS = {
    "QueryStartEvent", "RetrievalEndEvent", "EmbeddingEndEvent",
    "ReRankEndEvent", "LLMChatEndEvent", "LLMCompletionEndEvent",
    "GetResponseStartEvent",
}


def _check_event_drift() -> None:
    try:
        import importlib
        import pkgutil

        import llama_index.core.instrumentation.events as E  # type: ignore

        have = set()
        for m in pkgutil.walk_packages(E.__path__, E.__name__ + "."):
            try:
                mod = importlib.import_module(m.name)
            except Exception:
                continue
            have.update(n for n in dir(mod) if n.endswith("Event"))
        missing = _EXPECTED_EVENTS - have
        if missing:
            warn(f"LlamaIndex instrumentation is missing expected events {sorted(missing)} "
                 f"— version drift; those stages may not be captured")
    except Exception:
        pass


_REGISTERED_HANDLER = None


def _active_trace_id() -> str:
    """Share the registered event-handler's current trace so wrapper-logged
    events land in the same trace as retrieval/generation."""
    h = _REGISTERED_HANDLER
    if h is not None and h._state.get("trace_id"):
        return h._state["trace_id"]
    return RagEvent().trace_id


# Most rerankers (SBERTRerank/SentenceTransformerRerank, Cohere, LLMRerank, …)
# dispatch NO ReRank instrumentation event — only StructuredLLMRerank does. So
# the event handler can't see them. Wrap the postprocessor in a real
# BaseNodePostprocessor subclass (validates inside query engines) that logs a
# reranking event with before/after.
if _BasePostprocessor is not None:

    class _LoggedPostprocessor(_BasePostprocessor):  # type: ignore[misc, valid-type]
        target: Any = None
        model_config = {"arbitrary_types_allowed": True}

        @classmethod
        def class_name(cls) -> str:
            return "RagObservePostprocessor"

        def _postprocess_nodes(self, nodes, query_bundle=None):
            return self.target._postprocess_nodes(nodes, query_bundle)

        def postprocess_nodes(self, nodes, query_bundle=None, query_str=None):
            before = list(nodes or [])
            after = list(self.target.postprocess_nodes(
                nodes, query_bundle=query_bundle, query_str=query_str))
            try:
                ev = RagEvent(
                    trace_id=_active_trace_id(), project=_client.get_project(),
                    stage=Stage.RERANKING.value, name="llamaindex.reranker",
                    attributes={
                        "before": nodes_to_results(before),
                        "after": nodes_to_results(after),
                        "model": getattr(self.target, "model", None) or type(self.target).__name__,
                        "top_n": getattr(self.target, "top_n", None),
                    },
                )
                _client.get_client().log_event(ev.finish().model_dump())
            except Exception:
                pass
            return after


def instrument_postprocessor(postprocessor: Any) -> Any:
    """Wrap a LlamaIndex node postprocessor / reranker so ``postprocess_nodes``
    auto-logs a reranking event. Needed because most rerankers
    (SentenceTransformerRerank, Cohere, LLMRerank) emit no ReRank event."""
    if _BasePostprocessor is None:
        raise ImportError("LlamaIndex is not installed. Run: pip install ragobserve[llamaindex]")
    require_methods(postprocessor, ["postprocess_nodes"], "instrument_postprocessor")
    return _LoggedPostprocessor(target=postprocessor)
