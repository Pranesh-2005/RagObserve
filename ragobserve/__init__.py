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
from .client import flush, get_client, init
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

__version__ = "0.2.0"

__all__ = [
    "init", "flush", "get_client", "trace", "current_trace_id",
    "log_ingestion", "log_chunks", "log_embedding", "log_retrieval", "log_fusion",
    "log_rerank", "log_context", "log_generation", "log_ground_truth",
    "instrument_chroma", "instrument_pinecone", "instrument_qdrant",
    "instrument_weaviate", "instrument_milvus", "log_pgvector",
    "instrument_splitter", "instrument_embeddings", "instrument_loader",
    "instrument_compressor",
    "RagEvent", "Chunk", "Stage", "__version__",
]
