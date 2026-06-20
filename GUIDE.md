# RAGObserve — Complete Guide

**Local-first observability, debugging, and evaluation for RAG systems. The "MLflow for RAG."**

This is the full reference. It is written so that **a coding agent (even a small model) can read one section and add the exact, correct instrumentation** — and so that a human can use it as a docs site. Every public function, every framework adapter, every vector-DB wrapper, and every CLI command is documented with copy-paste examples.

RAGObserve records the whole retrieval lifecycle into a single local SQLite file and gives you a dashboard. **No server, no account, no cloud required.**

---

## Table of contents

1. [What you get](#1-what-you-get)
2. [Install](#2-install)
3. [The mental model](#3-the-mental-model)
4. [Initialize (once at startup)](#4-initialize-once-at-startup)
5. [Two ways to instrument — pick ONE per pipeline](#5-two-ways-to-instrument--pick-one-per-pipeline)
6. [Way A — Manual SDK (works with any framework)](#6-way-a--manual-sdk-works-with-any-framework)
7. [Logger reference (every `log_*` function)](#7-logger-reference-every-log_-function)
8. [Way B — Framework adapters (auto-capture)](#8-way-b--framework-adapters-auto-capture)
   - [LangChain](#81-langchain)
   - [LlamaIndex](#82-llamaindex)
9. [Vector database wrappers](#9-vector-database-wrappers)
10. [Cost tracking](#10-cost-tracking)
11. [The CLI & dashboard](#11-the-cli--dashboard)
12. [Live generation replay](#12-live-generation-replay)
13. [Local vs server (HTTP) mode](#13-local-vs-server-http-mode)
14. [Recipes — full copy-paste pipelines](#14-recipes--full-copy-paste-pipelines)
15. [Gotchas & troubleshooting](#15-gotchas--troubleshooting)
16. [Quick decision tree (for agents)](#16-quick-decision-tree-for-agents)

---

## 1. What you get

RAGObserve captures, per query, the entire RAG pipeline and shows it as a **waterfall**, with dedicated views:

- **Query Explorer** — every query with latency, cost, retriever, model, chunk count
- **Trace waterfall** — the full pipeline of one query, stage by stage, with timings
- **Retrieval Explorer** — retrieved chunks with scores, ranks, source, text
- **Hybrid Search Explorer** — per-source results (BM25 vs vector) and the fused result
- **Reranker Analytics** — before/after ordering with rank-shift arrows
- **Context Builder Viewer** — the *exact* prompt + injected chunks the model saw
- **Chunk Explorer** — most-retrieved / never-retrieved (dead) / duplicate chunks
- **Metrics** — Precision@k, Recall@k, MRR, nDCG (from logged ground truth) + chunk utilization
- **Generations & cost** — per-model / per-day token & dollar breakdowns, plus the captured context per generation
- **Live replay** — re-run any captured context against a live LLM provider, logged back into the trace

It is **framework-agnostic, provider-agnostic, and vector-DB-agnostic**.

---

## 2. Install

```bash
pip install ragobserve                 # core (fastapi, uvicorn, jinja2, pydantic, httpx)
pip install ragobserve[langchain]      # + LangChain adapter
pip install ragobserve[llamaindex]     # + LlamaIndex adapter
pip install ragobserve[dev]            # + pytest (for contributing)
```

- Vector-DB wrappers and the live-replay provider layer need **no** extra dependencies (they use `httpx` + duck typing).
- Python 3.9+.

---

## 3. The mental model

Every RAG system, regardless of framework, flows through the same lifecycle. RAGObserve records each step as a `RagEvent` tagged with a `stage`:

```
ingestion → chunking → embedding → indexing → retrieval → fusion
→ reranking → context_assembly → generation → grounding
```

Two concepts:

- **Trace** = one query / one run. (e.g. a user asks a question.)
- **Event** = one stage inside that trace. (e.g. the retrieval step.)

The dashboard reconstructs the waterfall from the events that share a trace.

You do **not** have to log every stage. Log what you have; the dashboard shows what it gets.

---

## 4. Initialize (once at startup)

Call `ragobserve.init()` **once**, before any logging.

```python
import ragobserve

ragobserve.init(project="my-rag")     # local store: hidden ./.ragobserve/ragobserve.db
```

Options:

```python
# Server mode (multiple processes, or remote dashboard):
ragobserve.init(project="my-rag", tracking_uri="http://localhost:5601")

# Custom database path:
ragobserve.init(project="my-rag", db_path="/abs/path/store.db")
```

| Argument | Default | Meaning |
|---|---|---|
| `project` | `"default"` | Logical project name; groups traces in the dashboard |
| `tracking_uri` | `None` | If set, send events over HTTP to a running server. If unset, write directly to local SQLite |
| `db_path` | `None` | Local mode only. Defaults to hidden `./.ragobserve/ragobserve.db` |

Notes:
- No `tracking_uri` → writes straight to SQLite, synchronous, nothing to flush.
- A legacy visible `./ragobserve.db` is auto-migrated into `./.ragobserve/` on first use.
- The folder is hidden (like `.git`).

---

## 5. Two ways to instrument — pick ONE per pipeline

| | Way A: Manual SDK | Way B: Framework adapter |
|---|---|---|
| Best for | Custom / hand-written pipelines, any framework | LangChain or LlamaIndex apps |
| How | Wrap query in `trace()`, call `log_*` | Attach a handler / call `register()` |
| Trace id | Managed by a contextvar | Managed by the adapter |

> **CRITICAL RULE: do not mix A and B inside one logical query.** The adapter manages its own trace id; the SDK loggers use a contextvar trace. Mixing them splits one query across two traces. Choose one per pipeline.

---

## 6. Way A — Manual SDK (works with any framework)

Wrap a query in `ragobserve.trace(...)`, then call loggers inside. All loggers attach to the active trace automatically.

```python
import ragobserve
ragobserve.init(project="my-rag")

with ragobserve.trace("query", query=question):
    ragobserve.log_retrieval(question, results, retriever="qdrant", top_k=5, duration_ms=23)
    ragobserve.log_fusion(fused, inputs={"bm25": bm25_hits, "vector": vec_hits}, strategy="rrf")
    ragobserve.log_rerank(before, after, model="bge-reranker", top_n=3)
    ragobserve.log_context(final_prompt, query=question, chunks=top_chunks, context_window=8192)
    ragobserve.log_generation(model="gpt-4o", prompt=final_prompt, response=answer,
                              input_tokens=812, output_tokens=197)   # cost auto-filled
```

### `trace` — three forms

```python
# 1. context manager (most common)
with ragobserve.trace("query", query=question):
    ...

# 2. decorator
@ragobserve.trace
def retrieve(query):
    ...

# 3. nested (parent/child spans, automatic via contextvars)
with ragobserve.trace("query", query=q):
    with ragobserve.trace("retrieve"):
        ...
```

`trace(name)` infers a stage from the name when it can (e.g. `"retrieval"`, `"rerank"`, `"generate"`); otherwise the span is tagged `other`. Pass extra keyword args (e.g. `query=...`) and they are stored as attributes.

### What "results" / "chunks" can be

The `results=` and `chunks=` arguments accept a **list** of any of these (mixed is fine):

- a plain `str` → becomes `{"text": "..."}`
- a `dict` with any of `{id, text, score, rank, source, metadata}`
- a `ragobserve.Chunk`
- a **LangChain or LlamaIndex document object** (anything with `.page_content`/`.text` and `.metadata`)

RAGObserve coerces them to the wire shape for you. To get **scores and source** to show in the dashboard, pass dicts (or doc objects), not bare strings:

```python
results = [
    {"text": "...", "score": 0.82, "source": "contract.pdf", "id": "c12"},
    {"text": "...", "score": 0.71, "source": "contract.pdf"},
]
ragobserve.log_retrieval(question, results, retriever="qdrant")
```

---

## 7. Logger reference (every `log_*` function)

All are top-level: `ragobserve.log_*`. All attach to the active `trace()`.

| Function | Required | Optional | Stage |
|---|---|---|---|
| `log_ingestion` | — | `source=`, `count=`, `sources=[...]` | ingestion |
| `log_chunks` | `chunks` | `strategy=`, `chunk_size=`, `overlap=` | chunking |
| `log_embedding` | `model` | `input_count=`, `dimensions=`, `cost=`, `duration_ms=` | embedding |
| `log_retrieval` | `query`, `results` | `top_k=`, `retriever=`, `duration_ms=` | retrieval |
| `log_fusion` | `results` | `inputs={name: [...]}`, `strategy=` | fusion |
| `log_rerank` | `before`, `after` | `model=`, `top_n=`, `duration_ms=` | reranking |
| `log_context` | `final_prompt` | `query=`, `system_prompt=`, `chunks=`, `context_window=` | context_assembly |
| `log_generation` | `model` | `prompt=`, `response=`, `input_tokens=`, `output_tokens=`, `cost=`, `duration_ms=` | generation |
| `log_ground_truth` | `relevant_chunk_ids=[...]` | `trace_id=` | (for metrics) |

### Signatures & examples

```python
# Ingestion — you loaded N source documents
ragobserve.log_ingestion(source="contract.pdf", count=1, sources=["contract.pdf"])

# Chunking — you split docs into chunks
ragobserve.log_chunks(chunks=[c.text for c in nodes],
                      strategy="sentence_splitter", chunk_size=512, overlap=50)

# Embedding — you embedded chunks (cost auto-filled if model known)
ragobserve.log_embedding(model="text-embedding-3-small", input_count=42, dimensions=1536)

# Retrieval — the core event
ragobserve.log_retrieval(query=question,
                         results=[{"text": "...", "score": 0.8, "source": "a.pdf"}],
                         retriever="qdrant", top_k=5, duration_ms=23)

# Fusion — hybrid search (per-source inputs + the fused list)
ragobserve.log_fusion(results=fused,
                      inputs={"bm25": bm25_hits, "vector": vec_hits},
                      strategy="rrf")

# Rerank — show before/after ordering
ragobserve.log_rerank(before=retrieved, after=reranked,
                      model="bge-reranker-base", top_n=3)

# Context assembly — exactly what the model will see
ragobserve.log_context(final_prompt=prompt, query=question,
                       system_prompt=sys_prompt, chunks=top_chunks,
                       context_window=8192)

# Generation — model + tokens → cost auto-backfilled
ragobserve.log_generation(model="gpt-4o", prompt=prompt, response=answer,
                          input_tokens=812, output_tokens=197)

# Ground truth — to compute Precision/Recall@k, MRR, nDCG.
# Call INSIDE the same trace; chunk ids must match retrieval result ids.
ragobserve.log_ground_truth(relevant_chunk_ids=["c12", "c19"])
```

Helpers also exported:

- `ragobserve.current_trace_id()` — the active trace id (or `None`).
- `ragobserve.flush()` — flush buffered events (only meaningful in server/HTTP mode).
- `ragobserve.Chunk`, `ragobserve.RagEvent`, `ragobserve.Stage` — model types if you want them.

> **Note on `log_generation`:** the `prompt=` you pass is what shows under "Context / prompt used". Pass the **full assembled prompt** (with the injected chunks), not the bare user question — otherwise the dashboard's context view will look empty. See [Gotchas](#15-gotchas--troubleshooting).

---

## 8. Way B — Framework adapters (auto-capture)

### 8.1 LangChain

```python
from ragobserve.adapters import (
    RagObserveCallbackHandler,
    instrument_loader, instrument_splitter,
    instrument_embeddings, instrument_compressor,
)
```

**Query time** — pass the handler; retrieval, context_assembly, and generation (model, tokens, cost) are captured automatically:

```python
# Use a FRESH handler per query → one trace per query
chain.invoke(question, config={"callbacks": [RagObserveCallbackHandler()]})
```

**Ingest time** — loaders, splitters, and embeddings emit **no** callbacks, so wrap them:

```python
loader   = instrument_loader(PyPDFLoader("contract.pdf"))     # → ingestion
splitter = instrument_splitter(RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50))  # → chunking
emb      = instrument_embeddings(OpenAIEmbeddings())          # → embedding (real Embeddings subclass; FAISS-safe)

docs   = loader.load()                     # logs ingestion (count + sources)
chunks = splitter.split_documents(docs)    # logs chunking
store  = FAISS.from_documents(chunks, emb) # embed_documents → logs embedding
```

**Reranking** — `compress_documents` fires no callback, so wrap the compressor (it stays a real `BaseDocumentCompressor`, so `ContextualCompressionRetriever` still accepts it):

```python
reranker  = instrument_compressor(CrossEncoderReranker(model=...))   # → reranking before/after
retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base)
```

What each LangChain helper logs:

| Helper | Wraps | Logs on | Stage |
|---|---|---|---|
| `RagObserveCallbackHandler()` | passed to `.invoke(config={"callbacks":[...]})` | retriever end, llm end, chain end | retrieval, context_assembly, generation, boundary |
| `instrument_loader(loader)` | `BaseLoader` | `.load()`, `.load_and_split()` | ingestion |
| `instrument_splitter(splitter)` | `TextSplitter` | `.split_documents/.split_text/.create_documents/.transform_documents` | chunking |
| `instrument_embeddings(emb)` | `Embeddings` | `.embed_documents()` (query embeds pass through) | embedding |
| `instrument_compressor(comp)` | `BaseDocumentCompressor` | `.compress_documents()` | reranking |

Notes:
- Create a **new `RagObserveCallbackHandler()` per query** so each query is its own trace.
- The handler reads token usage from both `llm_output` and chat-message `usage_metadata`.
- The handler emits **context_assembly** automatically (the prompt sent to the model is the assembled context) — no manual `log_context` needed.
- `instrument_*` return real subclasses/proxies — pass them exactly where the originals went.

### 8.2 LlamaIndex

**One call instruments the global dispatcher — ingest and query, all stages, no other code changes.**

```python
import ragobserve
from ragobserve.adapters.llamaindex import register, instrument_postprocessor

ragobserve.init(project="my-rag")
register()      # ← attach BEFORE building your index / query engine
```

Captured automatically after `register()`:

| Stage | Source event | Notes |
|---|---|---|
| embedding | `EmbeddingEndEvent` (incl. sparse) | model + dimensions |
| chunking | derived from the ingest embedding batch | LlamaIndex emits no node-parsing event |
| retrieval | `RetrievalEndEvent` | at the retriever layer → **all 80+ vector stores covered transitively** |
| context_assembly | `GetResponseStartEvent` | the exact context handed to the LLM |
| generation | `LLMChatEndEvent` / `LLMCompletionEndEvent` | model, prompt, response, tokens → cost |
| boundaries | `QueryStart/End`, `StreamChat*`, `AgentChatWithStep*` | de-duplicated against LLM events |

**Reranking** — most rerankers emit no event (`SentenceTransformerRerank`, Cohere, `LLMRerank`). Wrap them:

```python
reranker = instrument_postprocessor(SentenceTransformerRerank(top_n=3, model="..."))
query_engine = index.as_query_engine(node_postprocessors=[reranker])
```

That's it — `engine.query(question)` now produces a full trace. **Do not also call `log_retrieval` / `log_context` / `log_generation` yourself** — the adapter already captures them (doing both splits the trace, and your manual calls capture *less* than the adapter, e.g. the bare question instead of the real prompt).

Minimal LlamaIndex pipeline:

```python
import ragobserve
from ragobserve.adapters.llamaindex import register
from llama_index.core import VectorStoreIndex, Document

ragobserve.init(project="li-demo")
register()

index = VectorStoreIndex.from_documents([Document(text="...")])
engine = index.as_query_engine()
print(engine.query("what is the notice period?"))   # fully traced
# then: ragobserve ui
```

---

## 9. Vector database wrappers

Wrap a live client/collection once; every query becomes a retrieval event — no manual `log_retrieval`. Duck-typed: importing these never requires the DB package installed. Everything except the query method passes straight through.

```python
import ragobserve
ragobserve.init(project="my-rag")

col = ragobserve.instrument_chroma(chroma_collection)      # logs .query
idx = ragobserve.instrument_pinecone(pinecone_index)       # logs .query
qc  = ragobserve.instrument_qdrant(qdrant_client)          # logs .search / .query_points
wv  = ragobserve.instrument_weaviate(weaviate_collection)  # logs .query.near_vector/near_text/hybrid/bm25/fetch_objects
mv  = ragobserve.instrument_milvus(milvus_collection)      # logs .search (ORM Hit + MilvusClient dict)
```

| Function | Wraps | Intercepts | Optional |
|---|---|---|---|
| `instrument_chroma(collection)` | Chroma `Collection` | `.query(...)` | `retriever="chroma"` |
| `instrument_pinecone(index)` | Pinecone `Index` | `.query(...)` | `retriever="pinecone"` |
| `instrument_qdrant(client)` | Qdrant client | `.search`, `.query_points` | `retriever="qdrant"` |
| `instrument_weaviate(collection)` | Weaviate v4 `Collection` | `.query.near_vector/near_text/hybrid/bm25/fetch_objects` | `retriever="weaviate"` |
| `instrument_milvus(collection)` | pymilvus `Collection` / `MilvusClient` | `.search(...)` | `retriever="milvus"` |
| `log_pgvector(query, rows)` | — (no client to proxy) | you pass the fetched rows | see below |

pgvector has no client object to wrap — run your SQL, then pass the rows:

```python
cur.execute("SELECT text, source, embedding <=> %s AS distance FROM docs ORDER BY distance LIMIT 5", (vec,))
rows = cur.fetchall()
ragobserve.log_pgvector(query, rows)   # keys default to text/distance/source; override with text_key=, score_key=, source_key=
```

**Any** store works even without a dedicated wrapper — the `retriever` label is free text:

```python
ragobserve.log_retrieval(query, results, retriever="elasticsearch")   # or opensearch, faiss, ...
```

Use a wrapper **or** the LlamaIndex/LangChain adapter, not both for the same call (would double-log).

---

## 10. Cost tracking

Cost is **auto-backfilled** at ingest from the model id + token counts, using a built-in price book (`ragobserve/server/pricing.py`, USD per 1M tokens). So you usually only log `model` + `input_tokens` + `output_tokens`:

```python
ragobserve.log_generation(model="gpt-4o", input_tokens=812, output_tokens=197)  # cost computed for you
```

- Pass `cost=` to override the estimate.
- Unknown model → `cost=None` (no guessing). Add it to the `PRICE_BOOK` dict — it already includes Anthropic, OpenAI, Gemini, Groq/Llama, Mixtral, Mistral, DeepSeek, and local/ollama (=0).
- The dashboard's **Generations & cost** view shows per-model and per-day token & dollar breakdowns.

---

## 11. The CLI & dashboard

```bash
ragobserve ui          # start dashboard at http://127.0.0.1:5601
ragobserve providers   # list LLM providers and which have API keys set
ragobserve version     # print version
```

`ragobserve ui` flags / environment variables:

| Flag | Env var | Default |
|---|---|---|
| `--host` | `RAGOBSERVE_HOST` | `127.0.0.1` |
| `--port` | `RAGOBSERVE_PORT` | `5601` |
| `--backend-store-uri` | `RAGOBSERVE_STORE` | hidden `./.ragobserve/ragobserve.db` |

Example (bind all interfaces, custom port and DB — useful in containers):

```bash
RAGOBSERVE_HOST=0.0.0.0 RAGOBSERVE_PORT=8080 ragobserve ui --backend-store-uri /data/store.db
```

Run the UI from the same working directory as your app (so it finds `./.ragobserve/ragobserve.db`), or point `--backend-store-uri` at the file.

---

## 12. Live generation replay

From any trace's **Generation** or **Context** view, re-run an LLM over the *captured context* — to debug grounding, compare models, or fill in a generation that was never logged. The new generation is logged back into the same trace with its cost.

Under the hood: `POST /api/generate {provider, model, trace_id, max_tokens}`.

**11 providers** (all via `httpx`, detected at runtime from env vars):

| Provider id | Env var(s) | Default model |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-4-8` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-2.0-flash` |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| `openrouter` | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` |
| `together` | `TOGETHER_API_KEY` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| `mistral` | `MISTRAL_API_KEY` | `mistral-small-latest` |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| `fireworks` | `FIREWORKS_API_KEY` | `accounts/fireworks/models/llama-v3p3-70b-instruct` |
| `perplexity` | `PERPLEXITY_API_KEY` | `sonar` |
| `ollama` | — (local) | `llama3` (set `OLLAMA_HOST`, default `http://localhost:11434`) |

A provider is "ready" when its key is set (or, for Ollama, always). Check with `ragobserve providers`.

---

## 13. Local vs server (HTTP) mode

| | Local (default) | Server / HTTP |
|---|---|---|
| Set via | `init(project=...)` | `init(project=..., tracking_uri="http://host:5601")` |
| Writes | directly to SQLite, synchronous | batched on a background thread, POSTed to the server |
| When | single process / one app | many processes / app and dashboard on different hosts |
| Flush | not needed | automatic on exit; call `ragobserve.flush()` to force |

In HTTP mode the client batches events (100 per batch, every 1s) and flushes on process exit. Logging never blocks or crashes your app — failures are swallowed.

Run a server (the same `ragobserve ui` process serves both dashboard and the ingest API), then point apps at it:

```bash
ragobserve ui --host 0.0.0.0 --port 5601
```
```python
ragobserve.init(project="my-rag", tracking_uri="http://that-host:5601")
```

---

## 14. Recipes — full copy-paste pipelines

### 14.1 Custom pipeline (manual SDK)

```python
import ragobserve
ragobserve.init(project="demo")

def answer(question, retriever, llm):
    with ragobserve.trace("query", query=question):
        hits = retriever.search(question, k=5)   # list of dicts with text/score/source
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

### 14.2 LangChain RAG (ingest + query)

```python
import ragobserve
from ragobserve.adapters import (
    RagObserveCallbackHandler, instrument_loader, instrument_splitter, instrument_embeddings,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

ragobserve.init(project="lc-demo")

# ingest
docs   = instrument_loader(PyPDFLoader("contract.pdf")).load()
chunks = instrument_splitter(RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)).split_documents(docs)
emb    = instrument_embeddings(OpenAIEmbeddings())
store  = FAISS.from_documents(chunks, emb)

# query (fresh handler each call)
chain = ...  # your RetrievalQA / LCEL chain over store.as_retriever()
chain.invoke("notice period?", config={"callbacks": [RagObserveCallbackHandler()]})
# then: ragobserve ui
```

### 14.3 LlamaIndex RAG

```python
import ragobserve
from ragobserve.adapters.llamaindex import register
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

ragobserve.init(project="li-demo")
register()   # one call, before building anything

docs   = SimpleDirectoryReader("./data").load_data()
index  = VectorStoreIndex.from_documents(docs)   # embedding + chunking captured
engine = index.as_query_engine(similarity_top_k=5)
engine.query("notice period?")                   # retrieval + context + generation captured
# then: ragobserve ui
```

### 14.4 Standalone vector DB (no framework)

```python
import ragobserve
ragobserve.init(project="vdb-demo")

qc = ragobserve.instrument_qdrant(qdrant_client)
hits = qc.search(collection_name="docs", query_vector=vec, limit=5)   # auto-logged retrieval
# build prompt + call your LLM, then optionally:
ragobserve.log_generation(model="gpt-4o", input_tokens=700, output_tokens=150)
```

---

## 15. Gotchas & troubleshooting

1. **Don't mix manual SDK and adapter handlers in one trace** → it splits one query into two traces. Pick one per pipeline.
2. **"I can't see the retrieved chunks in the context / prompt."** You likely passed the bare question to `log_generation(prompt=...)`, or called `log_context` from a *different* retrieval pass than the one the LLM actually used. Fix: pass the **full assembled prompt** (with chunks) to `log_context(final_prompt=...)` and `log_generation(prompt=...)`, or — better for LangChain/LlamaIndex — let the adapter capture it (it grabs the real prompt the model saw). With LlamaIndex, calling `engine.query()` after `register()` captures the true context; manual logging around it does not and conflicts.
3. **LlamaIndex has no node-parsing event** → chunking is derived from the ingest embedding batch.
4. **Most rerankers emit no event** (LlamaIndex `SentenceTransformerRerank`/Cohere/`LLMRerank`; LangChain compressors) → use `instrument_postprocessor` / `instrument_compressor`.
5. **Token usage is best-effort** — read from the provider's `usage`. If absent → `None` → no cost. Pass `input_tokens`/`output_tokens` (or `cost=`) yourself to be sure.
6. **Unknown model → no cost.** Add it to `PRICE_BOOK` in `ragobserve/server/pricing.py`.
7. **Scores/source show "—"** → you passed bare strings. Pass dicts/doc objects with `score` and `source` (or `metadata.source`).
8. **Dashboard is empty** → run `ragobserve ui` from the same directory as your app (so it finds `./.ragobserve/ragobserve.db`), or pass `--backend-store-uri`. In HTTP mode, make sure the server is running and `tracking_uri` points to it.
9. **Version drift is loud, not silent** — if an `instrument_*` target lacks the expected method, or a LlamaIndex event class vanished, you get a `RagObserveWarning` (`ragobserve._diag.RagObserveWarning`) instead of empty capture. Watch your logs.
10. **Observability never crashes the app** — logging failures are caught and swallowed by design.

---

## 16. Quick decision tree (for agents)

```
Are you using LlamaIndex?
  → yes: ragobserve.init(...); from ragobserve.adapters.llamaindex import register; register()
         wrap rerankers with instrument_postprocessor(...). DO NOT add manual log_* calls.

Are you using LangChain?
  → yes: ragobserve.init(...)
         query:  chain.invoke(q, config={"callbacks":[RagObserveCallbackHandler()]})  # fresh per query
         ingest: instrument_loader / instrument_splitter / instrument_embeddings
         rerank: instrument_compressor(...)
         DO NOT add manual log_* for stages the handler already covers.

Using a raw vector DB client, no framework?
  → instrument_chroma/pinecone/qdrant/weaviate/milvus(client)  (or log_pgvector for pgvector)
    then log_generation(model=..., input_tokens=..., output_tokens=...) for cost.

Fully custom pipeline?
  → with ragobserve.trace("query", query=q):
        log_retrieval(...); [log_fusion]; [log_rerank]; log_context(...); log_generation(...)
    pass dicts (text/score/source) so scores & sources render.

Always finish with:  ragobserve ui   →  http://127.0.0.1:5601
```

---

*RAGObserve is local-first: your traces stay in `./.ragobserve/ragobserve.db` on your machine unless you deliberately run a shared server.*
