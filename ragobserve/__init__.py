"""RAGObserve — local-first observability for RAG systems.

Quickstart::

    import ragobserve
    ragobserve.init(project="contract-rag")          # local ./ragobserve.db
    # or: ragobserve.init(project="contract-rag", tracking_uri="http://localhost:5601")

    with ragobserve.trace("query", query="What is the notice period?"):
        ragobserve.log_retrieval(query, results, retriever="qdrant")
        ragobserve.log_rerank(before, after, model="bge-reranker")
        ragobserve.log_context(final_prompt, system_prompt=sys, chunks=chunks)
        ragobserve.log_generation(model="gpt-4o", response=answer, cost=0.002)

Then ``ragobserve ui`` to explore the dashboard.
"""
from .adapters.langchain import (
    instrument_compressor,
    instrument_embeddings,
    instrument_loader,
    instrument_splitter,
)
from .adapters.vectordb import (
    instrument_chroma,
    instrument_milvus,
    instrument_pinecone,
    instrument_qdrant,
    instrument_weaviate,
    log_pgvector,
)
from .client import flush, get_client, init, serve
from .eval import evaluate_trace, score_answer_relevance, score_faithfulness
from .stores import BaseStore, FileStore, MultiStore, PostgresStore, S3Store, SQLiteStore
from .events import Chunk, RagEvent, Stage
from .tracing import (
    current_trace_id,
    log_chunks,
    log_context,
    log_embedding,
    log_fusion,
    log_generation,
    log_ground_truth,
    log_ingestion,
    log_rerank,
    log_retrieval,
    trace,
)

try:  # single source of truth is pyproject.toml — this drifted to 0.3.0 once already
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("ragobserve")
except Exception:
    __version__ = "unknown"

__all__ = [
    "init", "flush", "get_client", "serve", "trace", "current_trace_id",
    "score_faithfulness", "score_answer_relevance", "evaluate_trace",
    "BaseStore", "SQLiteStore", "PostgresStore", "FileStore", "S3Store", "MultiStore",
    "log_ingestion", "log_chunks", "log_embedding", "log_retrieval", "log_fusion",
    "log_rerank", "log_context", "log_generation", "log_ground_truth",
    "instrument_chroma", "instrument_pinecone", "instrument_qdrant",
    "instrument_weaviate", "instrument_milvus", "log_pgvector",
    "instrument_splitter", "instrument_embeddings", "instrument_loader",
    "instrument_compressor",
    "RagEvent", "Chunk", "Stage", "__version__",
]
