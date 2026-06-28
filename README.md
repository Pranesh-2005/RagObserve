# RAGObserve

> v0.5.0

**Local-first observability, debugging and evaluation for RAG systems. The MLflow for RAG.**

Unlike general LLM observability tools, RAGObserve focuses on the *retrieval lifecycle*:

```
documents → chunking → embedding → indexing → retrieval → fusion
→ reranking → context assembly → generation → grounding
```

Framework-agnostic. Provider-agnostic. Vector-DB-agnostic. Zero required config — defaults to a local SQLite file inside `./.ragobserve/` (like `.git`). Scale up to Postgres or any cloud storage when you're ready.

## Install

```bash
pip install ragobserve                   # core (SQLite, dashboard, all adapters)
pip install ragobserve[langchain]        # + LangChain auto-instrumentation
pip install ragobserve[llamaindex]       # + LlamaIndex auto-instrumentation
pip install ragobserve[postgres]         # + PostgreSQL backend
pip install ragobserve[files]            # + FileStore (S3, GCS, Azure, Drive, local)
pip install ragobserve[files] s3fs       # FileStore → Amazon S3
pip install ragobserve[files] gcsfs      # FileStore → Google Cloud Storage
pip install ragobserve[files] adlfs      # FileStore → Azure Blob / ADLS
pip install ragobserve[files] gdrivefs   # FileStore → Google Drive
```

## Quickstart

```python
import ragobserve

ragobserve.init(project="contract-rag")
# or: ragobserve.init(project="contract-rag", tracking_uri="http://localhost:5601")

with ragobserve.trace("query", query=question):
    ragobserve.log_retrieval(question, results, retriever="qdrant", duration_ms=23)
    ragobserve.log_rerank(before, after, model="bge-reranker")
    ragobserve.log_context(final_prompt, system_prompt=sys, chunks=top_chunks, context_window=8192)
    ragobserve.log_generation(model="gpt-4o", prompt=final_prompt, response=answer, cost=0.002)
```

Async pipelines work identically — `async with ragobserve.trace(...)` and all `log_*` functions are safe to call from async code without blocking the event loop.

Then explore:

```bash
ragobserve ui          # http://127.0.0.1:5601?key=<key>
ragobserve export --project my-rag --output traces.ndjson
ragobserve eval   --project my-rag --api-key gsk_...
ragobserve providers
ragobserve version
```

Or start the dashboard from Python:

```python
ragobserve.serve()                    # same as `ragobserve ui`
ragobserve.serve(port=8080)           # custom port
```

## Do I need to create tables or a database?

**No. Everything is created automatically.**

| Backend | What happens on `init()` |
|---|---|
| SQLiteStore (default) | `.ragobserve/ragobserve.db` is created with the full schema |
| PostgresStore | all tables are created via `CREATE TABLE IF NOT EXISTS` on first connect |
| FileStore / S3 / GCS / Azure | directories/buckets are created on first write |

You never run migrations or create schemas manually.

## Storage backends

RAGObserve ships three backends. Swap them via `store=` in `init()`.

### SQLiteStore (default)

Zero config. Local file. Full dashboard.

```python
ragobserve.init(project="dev")                              # default hidden path
ragobserve.init(project="dev", db_path="/data/store.db")   # custom path
ragobserve.init(project="dev", store=ragobserve.SQLiteStore("/data/store.db"))
```

### PostgresStore

Full read/write. Dashboard works. Best for team deployments and production.

```python
ragobserve.init(
    project="prod",
    store=ragobserve.PostgresStore("postgresql://user:pass@host:5432/dbname"),
)
```

Tables are auto-created on first connect. No migrations needed. Requires `pip install ragobserve[postgres]`.

### FileStore

Write-only JSONL. Works with any [fsspec](https://filesystem-spec.readthedocs.io)-compatible target: S3, GCS, Azure Blob, Google Drive, SFTP, or local. No SQL queries — use with `MultiStore` for a dashboard, or query offline with DuckDB / Athena / BigQuery.

```python
ragobserve.init(project="prod", store=ragobserve.FileStore("s3://my-bucket/rag-events/"))
ragobserve.init(project="prod", store=ragobserve.FileStore("gs://my-bucket/rag-events/"))
ragobserve.init(project="prod", store=ragobserve.FileStore("az://container/rag-events/"))
ragobserve.init(project="prod", store=ragobserve.FileStore("gdrive://My Drive/rag-events/"))
ragobserve.init(project="prod", store=ragobserve.FileStore("/local/archive/"))
```

### MultiStore

Fan-out writes to multiple backends. Reads come from the first backend that supports them (the primary). The canonical pattern: local dashboard + durable cloud archive.

```python
store = ragobserve.MultiStore([
    ragobserve.SQLiteStore(),                       # primary: dashboard reads
    ragobserve.FileStore("s3://my-bucket/events/"), # sink: durable archive
])
ragobserve.init(project="prod", store=store)
```

### Bring your own

Any object that implements `ingest_events(events)`, `set_ground_truth(...)`, and `close()` is a valid store:

```python
class MyStore:
    def ingest_events(self, events): ...
    def set_ground_truth(self, trace_id, project, ids): ...
    def close(self): ...

ragobserve.init(project="prod", store=MyStore())
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

## Docker

```bash
# Docker Compose (recommended)
docker compose up
# → http://localhost:5601?key=<printed-key>

# Or plain Docker
docker build -t ragobserve .
docker run -p 5601:5601 -v ragobserve_data:/data \
  -e RAGOBSERVE_API_KEY=mysecretkey ragobserve
```

Data persists in the `ragobserve_data` named volume. Pass `GROQ_API_KEY` to enable `ragobserve eval` inside the container.

> **Single worker:** the container runs one uvicorn worker by default — required for the WebSocket live feed.

## Auth

When running in server mode (`ragobserve ui` or `ragobserve.serve()`), the REST API and WebSocket are protected by an API key.

```bash
# Auto-generated on first start; printed in the console URL
ragobserve ui
# → Dashboard: http://127.0.0.1:5601?key=<key>

# Set your own key
RAGOBSERVE_API_KEY=mysecretkey ragobserve ui
```

Clients authenticate via:

```
Authorization: Bearer <key>
# or
X-Api-Key: <key>
```

The dashboard auto-reads the key from the `?key=` URL param on first load and stores it in localStorage.

## LLM evaluation (faithfulness & answer relevance)

Rate your RAG system's answers with LLM-as-judge metrics powered by Groq (fast, free tier).

```python
from ragobserve.eval import score_faithfulness, score_answer_relevance, evaluate_trace

# Single metrics
faith = score_faithfulness(answer="90 days.", context=["Notice period is 90 days."])
# → {"score": 0.97, "reason": "All claims directly supported by context."}

rel = score_answer_relevance(answer="90 days.", query="What is the notice period?")
# → {"score": 0.95, "reason": "Directly answers the query."}

# Score a full trace (loads answer, context, and query automatically)
result = evaluate_trace(trace_data, api_key="gsk_...")
# → {"faithfulness": {...}, "answer_relevance": {...}}
```

Requires `GROQ_API_KEY` env var (or pass `api_key=` explicitly). Uses `llama3-8b-8192` by default; pass `model=` to change.

```bash
# Batch-eval all traces in a project from the CLI
ragobserve eval --project my-rag --api-key gsk_...
```

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
loader   = instrument_loader(PyPDFLoader("contract.pdf"))
splitter = instrument_splitter(RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50))
emb      = instrument_embeddings(OpenAIEmbeddings())   # real Embeddings subclass — FAISS-safe
```

### LlamaIndex

```python
from ragobserve.adapters.llamaindex import register
register()   # ONE call instruments the global dispatcher — ingest + query
```

| Stage | LangChain | LlamaIndex |
|---|---|---|
| ingestion | `instrument_loader` | (via pipeline) |
| chunking | `instrument_splitter` | auto |
| embedding | `instrument_embeddings` | auto |
| retrieval | auto (callback) | auto |
| reranking | `instrument_compressor` (or `log_rerank`) | auto |
| context assembly | auto (handler) | auto |
| generation + cost | auto | auto |

## Vector database integrations

```python
import ragobserve
ragobserve.init(project="my-rag")

col = ragobserve.instrument_chroma(chroma_collection)
idx = ragobserve.instrument_pinecone(pinecone_index)
qc  = ragobserve.instrument_qdrant(qdrant_client)
wv  = ragobserve.instrument_weaviate(weaviate_collection)
mv  = ragobserve.instrument_milvus(milvus_collection)

# pgvector — no client to proxy, pass the rows:
rows = cur.fetchall()
ragobserve.log_pgvector(query, rows)
```

## Export traces

```bash
# Export all traces for a project to NDJSON (one trace+events per line)
ragobserve export --project my-rag --output traces.ndjson

# Works with Postgres too
ragobserve export --project my-rag \
  --backend-store-uri postgresql://user:pass@host:5432/ragobs \
  --output traces.ndjson
```

## Health endpoint

```
GET /health  →  {"status": "ok", "version": "0.4.0"}
```

No auth required — use for load balancer health checks and container readiness probes.

## Live feed (WebSocket)

The dashboard **Query Explorer** auto-refreshes when new events arrive via WebSocket. You can also connect directly:

```javascript
const ws = new WebSocket("ws://localhost:5601/ws/traces?key=<apikey>&project=my-rag");
ws.onmessage = e => {
  const msg = JSON.parse(e.data);
  if (msg.type === "event") console.log(msg.data);
  // msg.type === "ping" every ~30s (keepalive)
};
```

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
