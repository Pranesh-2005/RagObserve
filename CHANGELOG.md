# Changelog

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
