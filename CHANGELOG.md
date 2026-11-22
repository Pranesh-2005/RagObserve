# Changelog

## [Unreleased]

### Added
- **Ragas integration** (`ragobserve.ragas`). `ragobserve export --format ragas`
  writes one Ragas sample per trace (`user_input`, `retrieved_contexts`,
  `response`, `retrieved_context_ids`, plus `reference_context_ids` /
  `reference` when logged). `ragobserve eval --backend ragas` and
  `POST /api/eval/{id}?backend=ragas` score traces with Ragas and store the
  results in the existing faithfulness / relevance columns. The
  `ragobserve[ragas]` extra is only needed for scoring.
- The LLM judge runs on any replay provider: `--provider` / `?provider=` /
  `provider=`.
- **OpenTelemetry ingest** at `POST /v1/traces` (OTLP/HTTP). It maps OpenInference,
  OpenLLMetry and GenAI-semconv spans to retrieval / rerank / generation /
  embedding events. JSON works out of the box; protobuf needs the
  `ragobserve[otel]` extra.
- **`ragobserve test`**: golden-set regression for CI. It runs a JSONL question
  set through `--pipeline module:fn`, writes a project per run, and exits 1 when
  any `--min-*` threshold (recall, precision, MRR, nDCG, faithfulness, answer
  relevance) isn't met. A threshold with no data counts as a failure.
- **Failure triage**: `GET /api/triage`, a Metrics-page section and a badge on
  every trace naming the first stage that broke.
- **Compare page** and `GET /api/compare?a=&b=`: two projects side by side, with
  deltas coloured better or worse.
- **Evaluation panel** on the trace page: shows stored scores and has a "Score this
  trace" button (LLM judge on any provider, or Ragas).
- **`ragobserve import-ragas`**: attaches scores from a Ragas results CSV/JSONL to
  traces, matched by `trace_id` (now included in `export --format ragas`) or by
  query.
- **Vision cost**: `log_generation(images=[{width, height}])` adds image tokens
  using Claude / Gemini / OpenAI sizing rules. SQLite and Postgres now share one
  cost-backfill function.
- **Replay keeps images**: the trace's image context is sent to OpenAI-compatible
  providers, Anthropic and Ollama.
- **`log_grounding`** and a Grounding panel (per-claim supported / unsupported,
  with evidence ids). A Chunking panel replaces the raw JSON view (so image
  chunks render too).

### Fixed
- **`@ragobserve.trace` on an `async def` logged a ~0 ms span.** The wrapper was
  sync, so the span closed when the coroutine was created, before any awaited
  work. Coroutine functions now get an async wrapper.
- **`ragobserve ui --backend-store-uri postgresql://...` opened SQLite.** `ui`
  now routes DSNs to `PostgresStore`, like `export` and `eval` do.
- **Re-sent events inflated chunk stats.** `chunk_retrievals` had no unique key,
  so a retried event added duplicate rows. Migration 2 (SQLite and Postgres)
  removes existing duplicates and adds a unique `(event_id, content_hash)` index.
- **The default eval model (`llama3-8b-8192`) is retired on Groq.** The judge now
  routes through the provider registry and uses the provider's default model.
  Groq has since retired every Llama chat model, so the Groq default is now
  `openai/gpt-oss-20b`, and `examples/test_rag.py`, `test_langchain.py` and
  `test_llamaindex.py`, which 404'd, use it too.
- **The judge's answer was cut off on reasoning models.** `max_tokens` went from
  150 to 1024; a trivial JSON reply already used 121 tokens on `gpt-oss-20b`.

### Fixed (found by running `examples/test_rag.py` end to end)
- **`ragobserve test` reported "no data" for retrieval when the LLM call failed.**
  Ground truth was logged after the pipeline returned. It's now logged first.
- **Triage said `retrieval_miss` on traces with recall 1.0.** The app logs int ids
  on retrieval and none on rerank/context. Triage now matches on id or text hash,
  and an item that has both links the two.
- **The trace query showed an inner span's rewritten query** instead of the
  user's question. The root span's query now wins.
- **`import-ragas` overwrote valid scores with `None`** when Ragas returned NaN
  for a metric. Empty and NaN cells are skipped now.
- **Replay generation always returned 401 from the dashboard.** The button's
  `fetch` sent no `Authorization` header. It now goes through a shared `apiPost`.
- **`ragobserve ui` crashed on Windows when its output was piped or redirected.**
  cp1252 can't encode the `→` it prints. stdout now replaces characters it
  can't encode.
- The S3 integration tests skip instead of failing when the bucket can't be
  reached with the current AWS credentials.

## [0.7.0] — 2026-07-30

### Added
- **Multimodal retrieval renders in the dashboard.** Image results used to show a
  blank Text cell — the row had a score and a source but nothing to look at, so a
  CLIP leg was indistinguishable from a scoring bug. Tag a result and the
  Retrieval Explorer, Hybrid Search Explorer and Reranker Analytics all show the
  thumbnail:
  ```python
  ragobserve.log_retrieval(query, [
      {"id": "img:7", "text": "", "score": 0.34, "source": "deck.pdf p.4",
       "metadata": {"modality": "image", "path": "/data/images/7.png"}},
  ], retriever="clip-ViT-B-32")
  ```
  `metadata.path` is served by the new `GET /api/image`; `metadata.image_b64`
  (plus optional `metadata.mime`) inlines the bytes instead, for remote stores.
  Untagged results render exactly as before.
- `GET /api/image?trace_id=&path=` — serves an image **only** if that trace logged
  that path, and only for known image extensions. The trace's own logged paths are
  the entire allowlist: without that check the endpoint would be an arbitrary
  local-file read for anyone who can reach the dashboard. Thumbnails are fetched
  with the `Authorization` header rather than a `?key=` URL, so the API key stays
  out of browser history, referrers and access logs.

### Notes
- Ranking metrics (Precision/Recall@k, MRR, nDCG, `chunk_utilization`) key off
  result `id`, so they already work across modalities — give image results stable
  ids distinct from your chunk ids.
- Vision **cost** is still text-only: `log_generation` takes `input_tokens`, and
  the price book has no per-image or per-tile rate. An image-heavy call
  underreports. Pass a corrected `cost=` if you need it exact.

## [0.6.0] — 2026-07-28

### Fixed
- **Instrumentation latency cut ~138x.** SQLite opened with the default
  `journal_mode=delete` + `synchronous=FULL`, so every `log_*` call paid its own
  fsync. Now `WAL` + `synchronous=NORMAL`. Measured on Windows, 10-chunk
  retrieval + ~5KB context + generation:

  | Path | Before (median) | After (median) |
  |---|---|---|
  | `log_retrieval` | 96.6 ms | 1.1 ms |
  | `log_context` | 96.1 ms | 1.4 ms |
  | `log_generation` | 95.6 ms | 0.2 ms |
  | full 4-span trace (sync) | 379.8 ms | 2.8 ms |
  | full 4-span trace (async) | *raised TypeError* | 1.6 ms |

  Crash-safety against process death is retained; only the last commits are at
  risk on host power loss, which is the right trade for observability data
  sitting on an app's hot path.
- **`async with ragobserve.trace(...)` raised `TypeError`.** `_TraceHandle`
  defined `__aexit__` but not `__aenter__`, so the async usage the README has
  always documented could not run. Added `__aenter__`.
- **Model cost overcharged up to 16x on `-mini` / `-nano` models.** The price
  book's substring fallback returned the *first* matching key, so `gpt-4o-mini`
  resolved to `gpt-4o` ($2.50/$10.00 instead of $0.15/$0.60). Lookup now prefers
  the longest matching key.
- **`ragobserve version` printed `0.3.0` on every release since 0.3.0.**
  `__init__.__version__` was hardcoded and drifted from `pyproject.toml`; it is
  now read from package metadata, the same source `/health` uses.

### Added
- **Automatic price refresh** — `ragobserve prices --refresh` pulls the
  community-maintained [LiteLLM price feed](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
  (~3,100 chat models across ~80 providers) and caches it to
  `~/.ragobserve/prices.json`. The cached feed takes priority over the built-in
  book, so costs stay current without waiting on a RAGObserve release. No vendor
  is privileged — Anthropic, OpenAI, Google, xAI, Meta, Mistral, DeepSeek,
  Cohere, Amazon, Alibaba and the hosted open-weight providers all come from the
  same feed. Point `RAGOBSERVE_PRICE_FEED` at your own URL to override.
  ```bash
  ragobserve prices --refresh          # download latest
  ragobserve prices                    # show feed status
  ragobserve prices --model gpt-4o-mini
  ```
- **Built-in price book expanded** 32 → 73 models, covering Claude 5 / Opus 4.x,
  GPT-5 family, Gemini 2.x, Grok, Llama 4, Qwen, Command, Nova, Magistral and
  more. Still only the offline fallback — `--refresh` is the source of truth.
- `pricing.refresh()`, `pricing.feed_info()`, `pricing.cache_path()` for
  programmatic use.
- `_lookup` is now `lru_cache`d (~1 µs per resolved model on the ingest path).

## [0.5.0] — 2026-06-28

### Added
- **Docker support** — `Dockerfile` (python:3.11-slim, installs core + postgres + files extras), `docker-compose.yml` (named volume for data persistence, env var pass-through for `RAGOBSERVE_API_KEY` and `GROQ_API_KEY`), `.dockerignore`.
  ```bash
  docker compose up          # http://localhost:5601
  docker build -t ragobserve .
  docker run -p 5601:5601 -v ragobserve_data:/data \
    -e RAGOBSERVE_API_KEY=mykey ragobserve
  ```
- **Single-worker documented** — WebSocket live feed uses an in-process bus; multi-worker deployments must run one uvicorn worker. Noted in Guide.md and as a `ponytail:` comment in `app.py`.
- **Eval scores in Query Explorer** — Faith. and Relev. columns visible in the trace table (populated after `ragobserve eval`).
- **Structured logging** — `logging.getLogger("ragobserve")` warnings on bus publish failures and MultiStore backend failures (previously silent swallows).

## [0.4.0] — 2026-06-28

### Added
- **Auth** — API key protection on all `/api/*` routes and WebSocket (`RAGOBSERVE_API_KEY`). Bearer token + `X-Api-Key` header both accepted. Dashboard auto-reads key from `?key=` URL param.
- **LLM evaluation** — `score_faithfulness`, `score_answer_relevance`, `evaluate_trace` via Groq LLM-as-judge. `POST /api/eval/{trace_id}` + `GET /api/eval/{trace_id}`. `ragobserve eval` CLI command.
- **WebSocket live feed** — `GET /ws/traces?key=&project=`. Query Explorer auto-refreshes on new events. 30s keepalive ping.
- **Rate limiting** — 500/min ingest, 10/min generate, 20/min eval (via slowapi).
- **Health endpoint** — `GET /health → {"status": "ok", "version": "..."}`. No auth required.
- **Export CLI** — `ragobserve export --project <name> --output traces.ndjson`. Supports SQLite and PostgreSQL backends.
- **Eval scores in Query Explorer** — Faithfulness and answer relevance columns shown in the trace table (populated after running `ragobserve eval`).
- **Schema migrations** — `schema_versions` table tracks applied migrations. First migration adds `eval_scores` table.
- **tiktoken** — accurate token counting (`cl100k_base`) with `len//4` fallback when tiktoken not installed.
- **Structured logging** — `logging.getLogger("ragobserve")` warnings on bus publish failures and MultiStore backend failures (previously silent).

### Fixed
- Eval JSON parsing: try full `json.loads` before regex extraction; regex now uses `.*` (greedy, handles nested braces) instead of `[^}]+`.
- PostgreSQL export now correctly detected by `postgresql://` or `postgres://` DSN prefix in CLI.
- WebSocket auth: `accept()` called before auth check (WebSocket protocol requirement).

### Changed
- `ragobserve ui` URL now includes `?key=` so first open is authenticated automatically.

## [0.3.0] — 2026-06-01

### Added
- PostgreSQL backend (`PostgresStore`) with `ThreadedConnectionPool` and `_get_conn()` context manager.
- FileStore: S3, GCS, Azure, Google Drive, local JSONL via fsspec. `makedirs` wrapped for cloud targets.
- MultiStore: fan-out writes, primary-backend reads.
- Live generation replay: 11 providers (Anthropic, OpenAI, Gemini, Groq, OpenRouter, Together, Mistral, DeepSeek, Fireworks, Perplexity, Ollama) via httpx.
- Cost auto-backfill from built-in price book on ingest.
- LangChain and LlamaIndex framework adapters.
- Vector DB wrappers: Chroma, Pinecone, Qdrant, Weaviate, Milvus, pgvector.
- Dashboard: Query Explorer, trace waterfall, Chunk Explorer, Metrics, Generations & cost views.
- `ragobserve ui`, `ragobserve providers`, `ragobserve version` CLI commands.
