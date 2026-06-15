# RAGObserve — Agent Guide

Guide for an LLM coding agent to instrument a RAG system with **RAGObserve**. Local-first observability for RAG ("MLflow for RAG"). Framework-, provider-, and vector-DB-agnostic. Writes to a local SQLite file — no server, no account needed.

## Install

```bash
pip install ragobserve                 # core (fastapi, uvicorn, jinja2, pydantic, httpx)
pip install ragobserve[langchain]      # + LangChain adapter
pip install ragobserve[llamaindex]     # + LlamaIndex adapter
pip install ragobserve[dev]            # + pytest
```
Vector-DB wrappers + the live-replay provider layer need **no** extra deps (httpx + duck typing).

## Mental model

Every RAG system flows through one lifecycle. RAGObserve records each step as a `RagEvent` with a `stage`:

```
ingestion → chunking → embedding → indexing → retrieval → fusion
→ reranking → context_assembly → generation → grounding
```

A **trace** = one query/run; its events = the pipeline steps. The dashboard reconstructs the waterfall from these.

## Init (do this once at startup)

```python
import ragobserve
ragobserve.init(project="my-rag")                 # local store: hidden ./.ragobserve/ragobserve.db
# server mode (multiple processes / remote): 
# ragobserve.init(project="my-rag", tracking_uri="http://localhost:5601")
# custom path:
# ragobserve.init(project="my-rag", db_path="/abs/path/store.db")
```
- No `tracking_uri` → writes straight to SQLite (synchronous, no flush needed).
- A legacy visible `./ragobserve.db` is auto-migrated into `./.ragobserve/` on first use.

## Two ways to instrument

### A. Manual SDK (any framework / custom pipeline) — RECOMMENDED for custom code

Wrap a query in `ragobserve.trace(...)`, call loggers inside. All loggers attach to the active trace via a contextvar.

```python
with ragobserve.trace("query", query=question):
    ragobserve.log_retrieval(question, results, retriever="qdrant", top_k=5, duration_ms=23)
    ragobserve.log_fusion(fused, inputs={"bm25": bm25_hits, "vector": vec_hits}, strategy="rrf")
    ragobserve.log_rerank(before, after, model="bge-reranker", top_n=3)
    ragobserve.log_context(final_prompt, query=question, chunks=top_chunks, context_window=8192)
    ragobserve.log_generation(model="gpt-4o", prompt=final_prompt, response=answer,
                            input_tokens=812, output_tokens=197)   # cost auto-filled
```

Also: `@ragobserve.trace` decorator; `ragobserve.trace` nests (parent/child spans via contextvars).

#### Logger reference (all in `ragobserve.*`)
| Logger | Key args | Stage |
|---|---|---|
| `log_ingestion` | `source=`, `count=`, `sources=[]` | ingestion |
| `log_chunks` | `chunks`, `strategy=`, `chunk_size=`, `overlap=` | chunking |
| `log_embedding` | `model`, `input_count=`, `dimensions=`, `cost=`, `duration_ms=` | embedding |
| `log_retrieval` | `query`, `results`, `top_k=`, `retriever=`, `duration_ms=` | retrieval |
| `log_fusion` | `results`, `inputs={name: [...]}`, `strategy=` | fusion |
| `log_rerank` | `before`, `after`, `model=`, `duration_ms=` | reranking |
| `log_context` | `final_prompt`, `query=`, `system_prompt=`, `chunks=`, `context_window=` | context_assembly |
| `log_generation` | `model`, `prompt=`, `response=`, `input_tokens=`, `output_tokens=`, `cost=`, `duration_ms=` | generation |
| `log_ground_truth` | `relevant_chunk_ids=[...]` (for retrieval metrics) | — |

**`results` / `chunks` accept**: `str`, `dict` (`{id, text, score, rank, source, metadata}`), `ragobserve.Chunk`, or LangChain/LlamaIndex doc objects (`.page_content`/`.text` + `.metadata`) — `normalize_result` coerces them.

### B. Framework adapters (auto-capture)

**Do NOT mix A and B in the same logical trace.** The callback handler / event handler manage their own trace id; the SDK loggers use the contextvar trace — mixing splits one query across two traces. Pick one per pipeline.

#### LangChain
```python
from ragobserve.adapters import (
    RagObserveCallbackHandler, instrument_loader, instrument_splitter,
    instrument_embeddings, instrument_compressor,
)

# query-time: retrieval + context_assembly + generation (model/tokens/cost) — automatic
chain.invoke(q, config={"callbacks": [RagObserveCallbackHandler()]})

# ingest-time: these emit NO callbacks, so wrap them
loader   = instrument_loader(PyPDFLoader("doc.pdf"))               # → ingestion
splitter = instrument_splitter(RecursiveCharacterTextSplitter(...)) # → chunking
emb      = instrument_embeddings(OpenAIEmbeddings())               # → embedding (real Embeddings subclass, FAISS-safe)
reranker = instrument_compressor(CrossEncoderReranker(...))        # → reranking (real BaseDocumentCompressor)
```
- Create a **fresh `RagObserveCallbackHandler()` per query** → one trace per query.
- `instrument_*` return real subclasses/proxies; pass them where the originals went (e.g. wrapped compressor into `ContextualCompressionRetriever`).

#### LlamaIndex
```python
from ragobserve.adapters.llamaindex import register, instrument_postprocessor

register()   # ONE call → instruments the global dispatcher:
             # embedding, chunking, retrieval (all vector stores), context_assembly,
             # generation (+tokens+cost), query + chat-engine boundaries

# rerankers (SentenceTransformerRerank, Cohere, LLMRerank) emit NO event — wrap them:
reranker = instrument_postprocessor(SentenceTransformerRerank(...))
```
- `register()` returns the handler. For manual pipelines (no query engine) that loop, reset per call: `handler._state["trace_id"] = None`.

#### Vector-DB wrappers (standalone / non-framework)
```python
col = ragobserve.instrument_chroma(chroma_collection)      # .query
idx = ragobserve.instrument_pinecone(pinecone_index)       # .query
qc  = ragobserve.instrument_qdrant(qdrant_client)          # .search / .query_points
wv  = ragobserve.instrument_weaviate(weaviate_collection)  # .query.near_vector/near_text/hybrid/bm25
mv  = ragobserve.instrument_milvus(milvus_collection)      # .search
ragobserve.log_pgvector(query, cur.fetchall())             # pgvector: no client, pass rows
```
Each auto-logs a retrieval event on query; everything else passes through.

## Cost tracing (Langfuse-style)

Cost is **auto-backfilled** at ingest from the model id + token counts via the price book (`ragobserve/server/pricing.py`, USD per 1M tokens). So just log `model` + `input_tokens` + `output_tokens`.
- Pass `cost=` explicitly to override.
- Unknown model → `cost=None` (no guess). Add it to `PRICE_BOOK` — it's a plain editable dict (Anthropic, OpenAI, Gemini, Groq/Llama, Mixtral, Mistral, DeepSeek, ollama/local=0 already in).

## CLI

```bash
ragobserve ui            # dashboard at http://127.0.0.1:5601  (env: RAGOBSERVE_PORT/HOST/STORE)
ragobserve providers     # list LLM providers + which have API keys set
ragobserve version
```
Dashboard pages: Query Explorer, trace waterfall, Retrieval Explorer, Hybrid Search Explorer, Reranker Analytics, Context Builder Viewer, Chunk Explorer, Metrics (Precision/Recall@k, MRR, nDCG), Generations & cost.

## Live generation replay

From a trace's Generation/Context view, re-run an LLM over the captured context. `POST /api/generate {provider, model, trace_id, max_tokens}`. 11 providers via httpx (Anthropic, OpenAI, Gemini, Groq, OpenRouter, Together, Mistral, DeepSeek, Fireworks, Perplexity, Ollama); enabled when the provider's API key env var is set.

## Gotchas (read before instrumenting)

1. **Don't mix manual SDK and adapter handlers in one trace** → split traces. Choose one per pipeline.
2. **LlamaIndex has no node-parsing event** → chunking is derived from the ingest embedding batch. Token-count chunk metadata is text-only.
3. **Most rerankers emit no event** (LlamaIndex SBERT/Cohere/LLM; LangChain compressors) → use `instrument_postprocessor` / `instrument_compressor`.
4. **Token usage is best-effort** — read from provider `raw.usage` / `usage_metadata`; `None` if absent → no cost.
5. **Version drift is loud**: if an `instrument_*` target lacks the expected method, or a LlamaIndex event class vanished, you get a `RagObserveWarning` (`ragobserve._diag.RagObserveWarning`) instead of silent empty capture. Watch for it.
6. **Observability never crashes the app** — logging failures are swallowed.

## Minimal end-to-end (copy-paste)

```python
import ragobserve
ragobserve.init(project="demo")

def answer(question, retriever, llm):
    with ragobserve.trace("query", query=question):
        hits = retriever.search(question, k=5)
        ragobserve.log_retrieval(question, hits, retriever="faiss")
        prompt = build_prompt(question, hits)
        ragobserve.log_context(prompt, query=question, chunks=hits, context_window=8192)
        resp = llm.complete(prompt)
        ragobserve.log_generation(model="llama-3.1-8b-instant", prompt=prompt,
                                response=resp.text,
                                input_tokens=resp.usage.input_tokens,
                                output_tokens=resp.usage.output_tokens)
        return resp.text
# then: ragobserve ui
```
