"""Auto-instrument wrappers for popular vector databases.

Wrap a live client/collection once and every query is logged as a RAGObserve
``retrieval`` event automatically — no manual ``log_retrieval`` calls::

    import ragobserve
    from ragobserve.adapters.vectordb import instrument_chroma, instrument_pinecone, instrument_qdrant

    ragobserve.init(project="my-rag")

    col = instrument_chroma(chroma_collection)
    col.query(query_texts=["notice period?"], n_results=3)      # logged

    idx = instrument_pinecone(pinecone_index)
    idx.query(vector=vec, top_k=5, include_metadata=True)        # logged

    qc = instrument_qdrant(qdrant_client)
    qc.search(collection_name="docs", query_vector=vec, limit=5) # logged

The wrappers are transparent proxies: any attribute/method other than the
intercepted query call passes straight through to the wrapped object. They use
duck typing only — none of the vector-DB packages are imported, so importing
this module never requires them to be installed.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .._diag import require_methods
from ..tracing import log_retrieval


# --------------------------------------------------------------- result mappers

def _chroma_results(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chroma ``query`` returns dict-of-lists, one inner list per query; we log
    the first query's hits."""
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    out = []
    for i, _id in enumerate(ids):
        meta = (metas[i] if i < len(metas) else None) or {}
        out.append({
            "id": _id,
            "text": docs[i] if i < len(docs) else None,
            "score": dists[i] if i < len(dists) else None,
            "source": meta.get("source"),
            "metadata": meta,
        })
    return out


def _pinecone_results(res: Any) -> List[Dict[str, Any]]:
    matches = res.get("matches") if isinstance(res, dict) else getattr(res, "matches", None)
    out = []
    for m in matches or []:
        get = (lambda k: m.get(k)) if isinstance(m, dict) else (lambda k: getattr(m, k, None))
        meta = get("metadata") or {}
        out.append({
            "id": get("id"),
            "score": get("score"),
            "text": meta.get("text") or meta.get("page_content"),
            "source": meta.get("source"),
            "metadata": meta,
        })
    return out


def _weaviate_results(res: Any) -> List[Dict[str, Any]]:
    objs = getattr(res, "objects", None)
    if objs is None and isinstance(res, dict):
        objs = res.get("objects")
    out = []
    for o in objs or []:
        props = getattr(o, "properties", None)
        if props is None and isinstance(o, dict):
            props = o.get("properties")
        props = props or {}
        md = getattr(o, "metadata", None)
        score = getattr(md, "score", None) if md is not None else None
        if score is None and md is not None:
            score = getattr(md, "distance", None)
        uuid = getattr(o, "uuid", None)
        if uuid is None and isinstance(o, dict):
            uuid = o.get("uuid")
        out.append({
            "id": str(uuid) if uuid is not None else None,
            "text": props.get("text") or props.get("content") or props.get("page_content"),
            "score": score,
            "source": props.get("source"),
            "metadata": props,
        })
    return out


def _milvus_results(res: Any) -> List[Dict[str, Any]]:
    # search returns one hit-list per query vector; log the first query's hits.
    try:
        hits = res[0] if res else []
    except (TypeError, KeyError, IndexError):
        hits = res or []
    out = []
    for h in hits:
        if isinstance(h, dict):
            ent = h.get("entity") or {}
            out.append({
                "id": h.get("id"),
                "score": h.get("distance") if h.get("distance") is not None else h.get("score"),
                "text": ent.get("text") or ent.get("page_content") or ent.get("document"),
                "source": ent.get("source"),
                "metadata": ent,
            })
        else:
            def eget(k):  # pymilvus Hit.entity is dict-like
                try:
                    return h.entity.get(k)
                except Exception:
                    return None
            out.append({
                "id": getattr(h, "id", None),
                "score": getattr(h, "distance", None),
                "text": eget("text") or eget("page_content") or eget("document"),
                "source": eget("source"),
                "metadata": None,
            })
    return out


def _qdrant_results(points: Any) -> List[Dict[str, Any]]:
    # ``search`` returns a list[ScoredPoint]; ``query_points`` returns a response
    # object whose ``.points`` holds them.
    if points is not None and not isinstance(points, (list, tuple)):
        points = getattr(points, "points", points)
    out = []
    for p in points or []:
        get = (lambda k: p.get(k)) if isinstance(p, dict) else (lambda k: getattr(p, k, None))
        payload = get("payload") or {}
        out.append({
            "id": get("id"),
            "score": get("score"),
            "text": payload.get("text") or payload.get("page_content") or payload.get("document"),
            "source": payload.get("source"),
            "metadata": payload,
        })
    return out


def _log(query: str, results: List[Dict[str, Any]], top_k: Optional[int],
         retriever: str, duration_ms: float) -> None:
    # observability must never break the instrumented app
    try:
        log_retrieval(query, results, top_k=top_k, retriever=retriever, duration_ms=duration_ms)
    except Exception:
        pass


# ------------------------------------------------------------------- proxy base

class _Proxy:
    """Transparent attribute-forwarding proxy. Subclasses override the one
    query method they want to instrument; everything else delegates."""

    def __init__(self, target: Any, retriever: str):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_retriever", retriever)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_target"), name, value)


# ----------------------------------------------------------------------- Chroma

class _ChromaProxy(_Proxy):
    def query(self, *args, **kwargs):
        qt = kwargs.get("query_texts")
        query = (qt[0] if isinstance(qt, (list, tuple)) and qt else None) or "<embedding query>"
        t0 = time.time()
        res = self._target.query(*args, **kwargs)
        _log(query, _chroma_results(res), kwargs.get("n_results"),
             self._retriever, (time.time() - t0) * 1000.0)
        return res


def instrument_chroma(collection: Any, retriever: str = "chroma") -> Any:
    """Wrap a Chroma ``Collection`` so ``.query(...)`` is auto-logged."""
    require_methods(collection, ["query"], "instrument_chroma")
    return _ChromaProxy(collection, retriever)


# --------------------------------------------------------------------- Pinecone

class _PineconeProxy(_Proxy):
    def query(self, *args, **kwargs):
        t0 = time.time()
        res = self._target.query(*args, **kwargs)
        _log("<vector query>", _pinecone_results(res), kwargs.get("top_k"),
             self._retriever, (time.time() - t0) * 1000.0)
        return res


def instrument_pinecone(index: Any, retriever: str = "pinecone") -> Any:
    """Wrap a Pinecone ``Index`` so ``.query(...)`` is auto-logged."""
    require_methods(index, ["query"], "instrument_pinecone")
    return _PineconeProxy(index, retriever)


# ----------------------------------------------------------------------- Qdrant

class _QdrantProxy(_Proxy):
    def search(self, *args, **kwargs):
        t0 = time.time()
        res = self._target.search(*args, **kwargs)
        _log("<vector query>", _qdrant_results(res), kwargs.get("limit"),
             self._retriever, (time.time() - t0) * 1000.0)
        return res

    def query_points(self, *args, **kwargs):
        t0 = time.time()
        res = self._target.query_points(*args, **kwargs)
        _log("<vector query>", _qdrant_results(res), kwargs.get("limit"),
             self._retriever, (time.time() - t0) * 1000.0)
        return res


def instrument_qdrant(client: Any, retriever: str = "qdrant") -> Any:
    """Wrap a Qdrant client so ``.search(...)`` / ``.query_points(...)`` are
    auto-logged."""
    require_methods(client, ["search", "query_points"], "instrument_qdrant")
    return _QdrantProxy(client, retriever)


# ---------------------------------------------------------------------- Weaviate

class _WeaviateQuery(_Proxy):
    """Wraps the v4 ``collection.query`` namespace (near_vector / near_text /
    hybrid / bm25 / fetch_objects)."""

    def _run(self, method, args, kwargs):
        q = kwargs.get("query")
        if not isinstance(q, str) and args and isinstance(args[0], str):
            q = args[0]
        query = q if isinstance(q, str) else "<vector query>"
        t0 = time.time()
        res = method(*args, **kwargs)
        _log(query, _weaviate_results(res), kwargs.get("limit"),
             self._retriever, (time.time() - t0) * 1000.0)
        return res

    def near_vector(self, *a, **k): return self._run(self._target.near_vector, a, k)
    def near_text(self, *a, **k): return self._run(self._target.near_text, a, k)
    def hybrid(self, *a, **k): return self._run(self._target.hybrid, a, k)
    def bm25(self, *a, **k): return self._run(self._target.bm25, a, k)
    def fetch_objects(self, *a, **k): return self._run(self._target.fetch_objects, a, k)


class _WeaviateProxy(_Proxy):
    @property
    def query(self):
        return _WeaviateQuery(self._target.query, self._retriever)


def instrument_weaviate(collection: Any, retriever: str = "weaviate") -> Any:
    """Wrap a Weaviate v4 ``Collection`` so every ``collection.query.*`` search
    (near_vector / near_text / hybrid / bm25 / fetch_objects) is auto-logged."""
    if getattr(collection, "query", None) is None:
        from .._diag import warn
        warn("instrument_weaviate: object has no .query namespace — nothing will be captured")
    return _WeaviateProxy(collection, retriever)


# ------------------------------------------------------------------------ Milvus

class _MilvusProxy(_Proxy):
    def search(self, *args, **kwargs):
        t0 = time.time()
        res = self._target.search(*args, **kwargs)
        _log("<vector query>", _milvus_results(res), kwargs.get("limit"),
             self._retriever, (time.time() - t0) * 1000.0)
        return res


def instrument_milvus(collection: Any, retriever: str = "milvus") -> Any:
    """Wrap a pymilvus ``Collection`` or ``MilvusClient`` so ``.search(...)`` is
    auto-logged (both ORM Hit results and MilvusClient dict results)."""
    require_methods(collection, ["search"], "instrument_milvus")
    return _MilvusProxy(collection, retriever)


# ---------------------------------------------------------------------- pgvector

def log_pgvector(query: str, rows: Any, text_key: str = "text",
                 score_key: str = "distance", source_key: str = "source",
                 retriever: str = "pgvector", top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Log a pgvector / Postgres similarity-search result.

    pgvector has no client object to proxy — you run raw SQL (``ORDER BY
    embedding <=> %s LIMIT k``) — so pass the query string and the fetched
    rows. Rows may be dicts (``RealDictCursor``) or plain tuples."""
    results = []
    for r in rows or []:
        if isinstance(r, dict):
            results.append({
                "text": r.get(text_key),
                "score": r.get(score_key),
                "source": r.get(source_key),
                "metadata": dict(r),
            })
        else:
            results.append({"text": str(r)})
    _log(query, results, top_k if top_k is not None else len(results), retriever, 0.0)
    return results
