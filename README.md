# RAGObserve

**Local-first observability, debugging and evaluation for RAG systems. The MLflow for RAG.**

Unlike general LLM observability tools, RAGObserve focuses on the *retrieval lifecycle*:

```
documents → chunking → embedding → indexing → retrieval → fusion
→ reranking → context assembly → generation → grounding
```

It is framework-agnostic (a universal RAG event model, not LangChain hooks), provider-agnostic, vector-DB-agnostic, and stores everything in a single local SQLite file inside a hidden `./.ragobserve/` folder (like `.git`) — no servers, no accounts.

## Install

```bash
pip install ragobserve            # or: uv tool install ragobserve
pip install ragobserve[langchain]   # optional LangChain auto-instrumentation
pip install ragobserve[llamaindex]  # optional LlamaIndex auto-instrumentation
```

## Quickstart

Instrument your RAG code (writes to a hidden `./.ragobserve/ragobserve.db`, no server needed):

```python
import ragobserve

ragobserve.init(project="contract-rag")
# or point at a running server:
# ragobserve.init(project="contract-rag", tracking_uri="http://localhost:5601")

with ragobserve.trace("query", query=question):
    ragobserve.log_retrieval(question, results, retriever="qdrant", duration_ms=23)
    ragobserve.log_rerank(before, after, model="bge-reranker")
    ragobserve.log_context(final_prompt, system_prompt=sys, chunks=top_chunks, context_window=8192)
    ragobserve.log_generation(model="gpt-4o", prompt=final_prompt, response=answer, cost=0.002)
```

Decorator and nesting also work:

```python
@ragobserve.trace
def retrieve(query): ...
```

Then explore:

```bash
ragobserve ui          # http://127.0.0.1:5601
```

## Dashboard

- **Query Explorer** — every query with latency, cost, retriever, model, chunk count
- **Trace waterfall** — the full pipeline per query, stage by stage
- **Retrieval Explorer** — retrieved chunks with scores, ranks, metadata
- **Hybrid Search Explorer** — BM25 vs vector vs fused results
- **Reranker Analytics** — before/after with rank shifts and Kendall's τ
- **Context Builder Viewer** — exactly what was sent to the model, DevTools-style
- **Chunk Explorer** — most retrieved / never retrieved (dead) / duplicate chunks
- **Metrics** — Precision@k, Recall@k, MRR, nDCG over logged ground truth, plus chunk utilization
- **Generations & cost** — Langfuse-style cost tracing: per-model / per-day token & $ breakdowns, charts, and the context that produced each generation. Costs are auto-backfilled from a built-in price book when you don't pass `cost=`.

## LLM generation & live replay

RAGObserve ships a zero-SDK, httpx-based provider layer covering **11 providers** — Anthropic, OpenAI, Gemini, Groq, OpenRouter, Together, Mistral, DeepSeek, Fireworks, Perplexity, Ollama. From any trace's **Generation** / **Context** view you can *replay* the captured context against a live provider (when its API key is set) and the new generation is logged back into the trace with its cost.

```bash
ragobserve providers   # list providers and which have keys configured
```

## Framework adapters

Full pipeline — ingest *and* query — is captured.

### LangChain

```python
from ragobserve.adapters import (
    RagObserveCallbackHandler,
    instrument_loader, instrument_splitter, instrument_embeddings,
)

# query-time: retrieval + generation (+ model, token usage, cost) via the handler
chain.invoke(q, config={"callbacks": [RagObserveCallbackHandler()]})

# ingest-time: loaders/splitters/embeddings emit no callbacks, so wrap them
loader   = instrument_loader(PyPDFLoader("contract.pdf"))            # → ingestion event
splitter = instrument_splitter(RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50))
emb      = instrument_embeddings(OpenAIEmbeddings())                 # real Embeddings subclass — FAISS-safe

docs   = loader.load()
chunks = splitter.split_documents(docs)   # → chunking event (split_documents/split_text/create_documents/transform_documents)
FAISS.from_documents(chunks, emb)         # embed_documents → embedding event
```

`instrument_embeddings` returns a true `Embeddings` subclass, so vector stores that `isinstance`-check it (FAISS, etc.) keep working; async `aembed_*` is covered via the base class. The callback handler reads token usage from both `llm_output` and chat-message `usage_metadata`. For reranking, `instrument_compressor(CrossEncoderReranker(...))` returns a real `BaseDocumentCompressor` subclass (so `ContextualCompressionRetriever` still validates it) and logs before/after on `compress_documents` — the one RAG step LangChain fires no callback for. The handler also emits **context_assembly** automatically (the prompt sent to the model is the assembled context — no manual `log_context` needed).

If a framework version moves an API the adapters hook, the wrappers emit a `RagObserveWarning` ("…not captured (version drift?)") instead of silently logging nothing.

### LlamaIndex

```python
from ragobserve.adapters.llamaindex import register
register()   # ONE call instruments the global dispatcher — ingest + query
```

Hooks LlamaIndex's instrumentation dispatcher, so it captures every stage with no code changes:

- **embedding** (`EmbeddingEndEvent`, incl. sparse) — model + dimensions
- **chunking** — derived from the ingest embedding batch (LlamaIndex emits no node-parsing event)
- **retrieval** (`RetrievalEndEvent`) — at the retriever layer, so **all 80+ vector stores** (Chroma/Pinecone/Qdrant/Milvus/Weaviate/…) are covered transitively
- **reranking** — `StructuredLLMRerank` fires `ReRankEndEvent` automatically; most rerankers (`SentenceTransformerRerank`, Cohere, `LLMRerank`) emit **no** event, so wrap them: `instrument_postprocessor(SentenceTransformerRerank(...))` → logs before/after, model, top_n
- **context_assembly** (`GetResponseStartEvent`) — the exact context handed to the LLM during synthesis
- **generation** (`LLMChat/CompletionEndEvent`) — model, prompt/response, tokens → **cost**
- **boundaries** — query engines (`QueryStart/End`) and chat engines (`StreamChat*`, `AgentChatWithStep*`, incl. streamed deltas), de-duplicated against the LLM events

| Stage | LangChain | LlamaIndex |
|---|---|---|
| ingestion | `instrument_loader` | (via pipeline) |
| chunking | `instrument_splitter` | auto |
| embedding | `instrument_embeddings` | auto |
| retrieval | auto (callback) | auto |
| reranking | `instrument_compressor` (or `log_rerank`) | auto |
| context assembly | auto (handler) | auto |
| generation + cost | auto | auto |
| query / chat boundary | auto (chain) | auto |

## Vector database integrations

Wrap a live client once; every query is logged as a retrieval event automatically — no manual `log_retrieval` calls. Duck-typed, so importing these never requires the DB package installed.

```python
import ragobserve
ragobserve.init(project="my-rag")

col = ragobserve.instrument_chroma(chroma_collection)     # .query
idx = ragobserve.instrument_pinecone(pinecone_index)      # .query
qc  = ragobserve.instrument_qdrant(qdrant_client)         # .search / .query_points
wv  = ragobserve.instrument_weaviate(weaviate_collection) # .query.near_vector/near_text/hybrid/bm25
mv  = ragobserve.instrument_milvus(milvus_collection)     # .search (ORM + MilvusClient)

# pgvector has no client to proxy — run your SQL, pass the rows:
rows = cur.fetchall()  # ORDER BY embedding <=> %s LIMIT k
ragobserve.log_pgvector(query, rows)
```

RAGObserve is vector-DB-agnostic: the `retriever` label is free-text, so **any** store works (FAISS, Elasticsearch, OpenSearch, pgvector, …) even without a dedicated wrapper — just pass results to `ragobserve.log_retrieval(query, results, retriever="...")`.

## Try the demo

```bash
python examples/demo_rag.py
ragobserve ui
```

## Development

```bash
pip install -e .[dev]
pytest
```
