"""Pluggable store backends for RAGObserve.

The default is SQLiteStore (zero-config, local file). Swap it out by passing
``store=`` to ``ragobserve.init()``::

    ragobserve.init(project="prod", store=ragobserve.PostgresStore("postgresql://..."))
    ragobserve.init(project="prod", store=ragobserve.FileStore("s3://bucket/prefix/"))
    ragobserve.init(project="prod", store=ragobserve.MultiStore([
        ragobserve.SQLiteStore(),            # dashboard reads from local DB
        ragobserve.FileStore("s3://bucket"), # durable archive
    ]))

Backend summary
---------------
SQLiteStore     default, zero-config                        (built-in)
PostgresStore   full read/write, team/prod use              pip install ragobserve[postgres]
FileStore       write-only JSONL, any fsspec URI            pip install ragobserve[files]
                  s3://bucket/prefix        → s3fs
                  gs://bucket/prefix        → gcsfs
                  az://container/prefix     → adlfs
                  gdrive://folder/path      → gdrivefs
                  /local/path/              → built-in
MultiStore      fan-out writes, primary-backend reads       (built-in)

Bring-your-own backend: implement BaseStore and pass it to init().
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Protocol, runtime_checkable

_log = logging.getLogger("ragobserve")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class BaseStore(Protocol):
    """Minimal interface every RAGObserve store backend must satisfy.

    Only ``ingest_events``, ``set_ground_truth``, and ``close`` are required
    for write-only / sink backends. The read methods are called by the
    dashboard server — omit them (or raise NotImplementedError) if reads are
    not supported.
    """

    def ingest_events(self, events: List[Dict[str, Any]]) -> int: ...
    def set_ground_truth(self, trace_id: str, project: str, relevant_chunk_ids: List[str]) -> None: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# SQLiteStore — default backend (thin alias to the server-side Store)
# ---------------------------------------------------------------------------

class SQLiteStore:
    """Zero-config SQLite store. Identical to the default local backend.

    Usage::

        ragobserve.init(project="dev", store=ragobserve.SQLiteStore("/path/to/my.db"))
    """

    def __new__(cls, db_path: Optional[str] = None):  # type: ignore[misc]
        from .server.db import Store
        from .storage import resolve

        return Store(resolve(db_path))


# ---------------------------------------------------------------------------
# FileStore — write-only JSONL via fsspec (S3, GCS, Azure, Drive, local, …)
# ---------------------------------------------------------------------------

class FileStore:
    """Write-only store that appends events as JSONL to any fsspec-compatible
    filesystem — S3, GCS, Azure Blob, Google Drive, SFTP, local, …

    Each event is written to::

        <root>/<project>/<YYYY-MM-DD>/<event_id>.jsonl

    Install the matching fsspec driver alongside ``ragobserve[files]``::

        pip install ragobserve[files] s3fs          # Amazon S3
        pip install ragobserve[files] gcsfs          # Google Cloud Storage
        pip install ragobserve[files] adlfs          # Azure Blob / ADLS
        pip install ragobserve[files] gdrivefs       # Google Drive
        pip install ragobserve[files]                # local filesystem

    Usage::

        # S3
        store = ragobserve.FileStore("s3://my-bucket/rag-events/")
        # GCS
        store = ragobserve.FileStore("gs://my-bucket/rag-events/")
        # Azure
        store = ragobserve.FileStore("az://my-container/rag-events/")
        # Google Drive
        store = ragobserve.FileStore("gdrive://My Drive/rag-events/")
        # Local
        store = ragobserve.FileStore("/data/rag-events/")

        ragobserve.init(project="prod", store=store)

    Dashboard reads are not supported — pair with a SQL backend via
    ``MultiStore``, or query the files offline with DuckDB / Athena / BigQuery.
    """

    def __init__(self, root: str, **storage_options: Any):
        try:
            import fsspec
        except ImportError:
            raise ImportError("FileStore requires fsspec: pip install ragobserve[files]")

        self._root = root.rstrip("/")
        self._fs = fsspec.filesystem(self._root.split("://")[0] if "://" in self._root else "file",
                                     **storage_options)

    def _path(self, project: str, event_id: str, start_time: Optional[float] = None) -> str:
        day = time.strftime("%Y-%m-%d", time.gmtime(start_time or time.time()))
        return f"{self._root}/{project}/{day}/{event_id}.jsonl"

    def ingest_events(self, events: List[Dict[str, Any]]) -> int:
        for ev in events:
            project = ev.get("project") or "default"
            event_id = ev.get("event_id") or ev.get("trace_id") or "unknown"
            path = self._path(project, event_id, ev.get("start_time"))
            try:
                self._fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
            except Exception:
                pass  # ponytail: S3/cloud has no real dirs; makedirs may fail at bucket level
            with self._fs.open(path, "wb") as f:
                f.write(json.dumps(ev, default=str).encode())
        return len(events)

    def set_ground_truth(self, trace_id: str, project: str, relevant_chunk_ids: List[str]) -> None:
        path = self._path(project, f"gt_{trace_id}")
        try:
            self._fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
        except Exception:
            pass
        body = json.dumps({"trace_id": trace_id, "project": project,
                           "relevant_chunk_ids": relevant_chunk_ids}).encode()
        with self._fs.open(path, "wb") as f:
            f.write(body)

    def close(self) -> None:
        pass

    def _not_supported(self, *_: Any, **__: Any) -> Any:
        raise NotImplementedError(
            "FileStore is write-only. For dashboard reads, combine with a SQL "
            "backend via MultiStore, or query the files offline with DuckDB."
        )
    _not_supported._is_not_supported = True  # type: ignore[attr-defined]

    list_projects = _not_supported
    list_traces = _not_supported
    get_trace = _not_supported
    chunk_views = _not_supported
    cost_summary = _not_supported
    list_generations = _not_supported
    get_generation_context = _not_supported
    traces_with_ground_truth = _not_supported


# convenience alias
def S3Store(bucket: str, prefix: str = "", region: Optional[str] = None, **kw: Any) -> "FileStore":
    """Convenience wrapper — returns a FileStore pointed at s3://bucket/prefix.

    For new code prefer ``FileStore("s3://bucket/prefix/")`` directly.
    Requires ``pip install ragobserve[files] s3fs``.
    """
    root = f"s3://{bucket}/{prefix.lstrip('/')}"
    opts: Dict[str, Any] = {}
    if region:
        opts["client_kwargs"] = {"region_name": region}
    opts.update(kw)
    return FileStore(root, **opts)


# ---------------------------------------------------------------------------
# PostgresStore — full read/write, mirrors SQLite schema
# ---------------------------------------------------------------------------

_PG_MIGRATIONS = [
    (1, """
CREATE TABLE IF NOT EXISTS eval_scores (
    trace_id     TEXT NOT NULL,
    project      TEXT NOT NULL,
    metric       TEXT NOT NULL,
    score        DOUBLE PRECISION,
    reason       TEXT,
    model        TEXT,
    evaluated_at DOUBLE PRECISION,
    PRIMARY KEY (trace_id, metric)
);
"""),
]

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    name TEXT PRIMARY KEY,
    created_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    project  TEXT NOT NULL,
    name     TEXT,
    query    TEXT,
    start_time DOUBLE PRECISION,
    end_time   DOUBLE PRECISION,
    status   TEXT DEFAULT 'ok'
);
CREATE INDEX IF NOT EXISTS idx_traces_project ON traces(project);
CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT PRIMARY KEY,
    trace_id      TEXT NOT NULL,
    span_id       TEXT,
    parent_span_id TEXT,
    project       TEXT NOT NULL,
    stage         TEXT NOT NULL,
    name          TEXT,
    start_time    DOUBLE PRECISION,
    end_time      DOUBLE PRECISION,
    duration_ms   DOUBLE PRECISION,
    status        TEXT DEFAULT 'ok',
    attributes    JSONB DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_trace         ON events(trace_id);
CREATE INDEX IF NOT EXISTS idx_events_project_stage ON events(project, stage);
CREATE TABLE IF NOT EXISTS chunks (
    project      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT '',
    text         TEXT,
    token_count  INTEGER,
    metadata     JSONB DEFAULT '{}',
    first_seen   DOUBLE PRECISION,
    PRIMARY KEY (project, content_hash, source)
);
CREATE TABLE IF NOT EXISTS chunk_retrievals (
    id           BIGSERIAL PRIMARY KEY,
    event_id     TEXT NOT NULL,
    trace_id     TEXT NOT NULL,
    project      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    score        DOUBLE PRECISION,
    rank         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_cr_project ON chunk_retrievals(project, content_hash);
CREATE INDEX IF NOT EXISTS idx_cr_trace   ON chunk_retrievals(trace_id);
CREATE TABLE IF NOT EXISTS ground_truth (
    trace_id           TEXT PRIMARY KEY,
    project            TEXT NOT NULL,
    relevant_chunk_ids JSONB NOT NULL
);
"""


class PostgresStore:
    """Full read/write store backed by PostgreSQL.

    Mirrors the SQLite schema with JSONB for attributes — the full
    RAGObserve dashboard works out of the box. Suitable for team
    deployments and production use.

    Requires ``pip install ragobserve[postgres]`` (psycopg2-binary).

    Usage::

        store = ragobserve.PostgresStore("postgresql://user:pass@host:5432/dbname")
        ragobserve.init(project="prod", store=store)

    The connection string is passed directly to psycopg2 — see
    https://www.psycopg.org/docs/module.html#psycopg2.connect for all options.
    """

    def __init__(self, dsn: str, min_conn: int = 1, max_conn: int = 10):
        try:
            import psycopg2.extras
            import psycopg2.pool
        except ImportError:
            raise ImportError(
                "PostgresStore requires psycopg2: pip install ragobserve[postgres]"
            )
        self._pool = psycopg2.pool.ThreadedConnectionPool(
            min_conn, max_conn, dsn,
            cursor_factory=psycopg2.extras.RealDictCursor,
        )
        self._apply_schema()

    @contextmanager
    def _get_conn(self) -> Generator:
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def _apply_schema(self) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                for stmt in _PG_SCHEMA.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        cur.execute(stmt)
                # schema migrations
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS schema_versions "
                    "(version INTEGER PRIMARY KEY, applied_at DOUBLE PRECISION)"
                )
                cur.execute("SELECT MAX(version) AS v FROM schema_versions")
                row = cur.fetchone()
                current = (row["v"] or 0) if row else 0
                for version, sql in _PG_MIGRATIONS:
                    if version > current:
                        for stmt in sql.strip().split(";"):
                            stmt = stmt.strip()
                            if stmt:
                                cur.execute(stmt)
                        cur.execute(
                            "INSERT INTO schema_versions(version, applied_at) "
                            "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (version, __import__("time").time()),
                        )

    def close(self) -> None:
        self._pool.closeall()

    # ---------------------------------------------------------------- ingest

    def ingest_events(self, events: List[Dict[str, Any]]) -> int:
        from psycopg2.extras import Json

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                for ev in events:
                    self._ingest_one(cur, ev, Json)
        return len(events)

    def _ingest_one(self, cur: Any, ev: Dict[str, Any], Json: Any) -> None:
        from .server import pricing
        from .events import content_hash, estimate_tokens

        project = ev.get("project") or "default"
        trace_id = ev.get("trace_id") or ev.get("event_id")
        attrs = dict(ev.get("attributes") or {})

        if ev.get("stage") == "generation" and attrs.get("cost") in (None, 0):
            est = pricing.estimate_cost(attrs.get("model"), attrs.get("input_tokens"),
                                        attrs.get("output_tokens"))
            if est is not None:
                attrs["cost"] = est
                attrs["cost_estimated"] = True

        cur.execute(
            "INSERT INTO projects(name, created_at) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (project, time.time()),
        )

        cur.execute(
            "SELECT trace_id, query, start_time, end_time, status FROM traces WHERE trace_id=%s",
            (trace_id,),
        )
        row = cur.fetchone()
        start = ev.get("start_time") or time.time()
        end = ev.get("end_time") or start
        query = attrs.get("query")
        name = ev.get("name") or ev.get("stage")

        if row is None:
            cur.execute(
                "INSERT INTO traces(trace_id, project, name, query, start_time, end_time, status)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (trace_id, project, name, query, start, end, ev.get("status", "ok")),
            )
        else:
            new_start = min(row["start_time"] or start, start)
            new_end = max(row["end_time"] or end, end)
            new_query = row["query"] or query
            status = "error" if (row["status"] == "error" or ev.get("status") == "error") else "ok"
            cur.execute(
                "UPDATE traces SET start_time=%s, end_time=%s, query=%s, status=%s WHERE trace_id=%s",
                (new_start, new_end, new_query, status, trace_id),
            )

        cur.execute(
            "INSERT INTO events(event_id, trace_id, span_id, parent_span_id, project, stage, name,"
            " start_time, end_time, duration_ms, status, attributes)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT (event_id) DO UPDATE SET"
            "   start_time=EXCLUDED.start_time, end_time=EXCLUDED.end_time,"
            "   duration_ms=EXCLUDED.duration_ms, status=EXCLUDED.status,"
            "   attributes=EXCLUDED.attributes",
            (
                ev.get("event_id"), trace_id, ev.get("span_id"), ev.get("parent_span_id"),
                project, ev.get("stage", "other"), ev.get("name", ""),
                ev.get("start_time"), ev.get("end_time"), ev.get("duration_ms"),
                ev.get("status", "ok"), Json(attrs),
            ),
        )
        self._extract_chunks(cur, project, trace_id, ev.get("event_id"),
                             ev.get("stage"), attrs, Json)

    def _extract_chunks(self, cur: Any, project: str, trace_id: str,
                        event_id: str, stage: str, attrs: Dict, Json: Any) -> None:
        from .events import content_hash, estimate_tokens

        def register(item: Dict[str, Any]) -> Optional[str]:
            text = item.get("text") or ""
            chash = item.get("id") or (content_hash(text) if text else None)
            if chash is None:
                return None
            cur.execute(
                "INSERT INTO chunks(project, content_hash, source, text, token_count, metadata, first_seen)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (
                    project, chash, item.get("source") or "", text,
                    item.get("token_count") or (estimate_tokens(text) if text else None),
                    Json(item.get("metadata") or {}), time.time(),
                ),
            )
            return chash

        if stage == "chunking":
            for item in attrs.get("chunks") or []:
                if isinstance(item, dict):
                    register(item)
        elif stage in ("retrieval", "fusion"):
            for i, item in enumerate(attrs.get("results") or []):
                if not isinstance(item, dict):
                    continue
                chash = register(item)
                if chash:
                    cur.execute(
                        "INSERT INTO chunk_retrievals(event_id, trace_id, project, content_hash, score, rank)"
                        " VALUES (%s,%s,%s,%s,%s,%s)",
                        (event_id, trace_id, project, chash,
                         item.get("score"), item.get("rank", i + 1)),
                    )
        elif stage == "reranking":
            for item in attrs.get("after") or []:
                if isinstance(item, dict):
                    register(item)
        elif stage == "context_assembly":
            for item in attrs.get("chunks") or []:
                if isinstance(item, dict):
                    register(item)

    def set_ground_truth(self, trace_id: str, project: str, relevant_chunk_ids: List[str]) -> None:
        from psycopg2.extras import Json

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ground_truth(trace_id, project, relevant_chunk_ids)"
                    " VALUES (%s,%s,%s)"
                    " ON CONFLICT (trace_id) DO UPDATE SET relevant_chunk_ids=EXCLUDED.relevant_chunk_ids",
                    (trace_id, project, Json(relevant_chunk_ids)),
                )

    # ---------------------------------------------------------------- reads
    # SQL is identical to SQLite except:
    #   json_extract(col,'$.x')  →  col->>'x'
    #   CAST(... AS numeric)     →  (col->>'x')::numeric
    #   ?                        →  %s
    #   attributes returned as dict by psycopg2, no json.loads needed

    def list_projects(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT p.name,
                           COUNT(t.trace_id) AS traces,
                           AVG((t.end_time - t.start_time) * 1000.0) AS avg_latency_ms,
                           MAX(t.end_time) AS last_activity
                    FROM projects p LEFT JOIN traces t ON t.project = p.name
                    GROUP BY p.name ORDER BY last_activity DESC
                    """
                )
                rows = cur.fetchall()
                out = []
                for r in rows:
                    cur.execute(
                        "SELECT SUM((attributes->>'cost')::numeric) AS c FROM events"
                        " WHERE project=%s AND stage IN ('generation','embedding')",
                        (r["name"],),
                    )
                    cost = cur.fetchone()
                    out.append({
                        "name": r["name"], "traces": r["traces"],
                        "avg_latency_ms": r["avg_latency_ms"],
                        "total_cost": float(cost["c"] or 0),
                        "last_activity": r["last_activity"],
                    })
        return out

    def list_traces(self, project: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        where = "WHERE t.project = %s" if project else ""
        args: list = [project] if project else []
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT t.*,
                      (SELECT SUM((e.attributes->>'cost')::numeric) FROM events e
                         WHERE e.trace_id = t.trace_id AND e.stage='generation') AS cost,
                      (SELECT e.attributes->>'model' FROM events e
                         WHERE e.trace_id = t.trace_id AND e.stage='generation' LIMIT 1) AS model,
                      (SELECT e.attributes->>'retriever' FROM events e
                         WHERE e.trace_id = t.trace_id AND e.stage='retrieval' LIMIT 1) AS retriever,
                      (SELECT COUNT(*) FROM chunk_retrievals c WHERE c.trace_id = t.trace_id) AS chunk_count
                    FROM traces t {where}
                    ORDER BY t.start_time DESC LIMIT %s
                    """,
                    args + [limit],
                )
                rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if r["end_time"] and r["start_time"]:
                d["duration_ms"] = (r["end_time"] - r["start_time"]) * 1000.0
            else:
                d["duration_ms"] = None
            out.append(d)
        return out

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM traces WHERE trace_id=%s", (trace_id,))
                t = cur.fetchone()
                if t is None:
                    return None
                cur.execute(
                    "SELECT * FROM events WHERE trace_id=%s ORDER BY start_time", (trace_id,)
                )
                evs = cur.fetchall()
                cur.execute("SELECT * FROM ground_truth WHERE trace_id=%s", (trace_id,))
                gt = cur.fetchone()
        events = [dict(e) for e in evs]  # attributes already a dict (JSONB)
        trace = dict(t)
        if t["end_time"] and t["start_time"]:
            trace["duration_ms"] = (t["end_time"] - t["start_time"]) * 1000.0
        return {
            "trace": trace,
            "events": events,
            "ground_truth": gt["relevant_chunk_ids"] if gt else None,
        }

    def chunk_views(self, project: str, view: str = "top", limit: int = 100) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                if view == "top":
                    cur.execute(
                        """
                        SELECT c.content_hash, MIN(c.source) AS source, MIN(c.text) AS text,
                               MIN(c.token_count) AS token_count,
                               COUNT(cr.id) AS retrievals, AVG(cr.score) AS avg_score
                        FROM chunks c JOIN chunk_retrievals cr
                          ON cr.project = c.project AND cr.content_hash = c.content_hash
                        WHERE c.project = %s
                        GROUP BY c.content_hash ORDER BY retrievals DESC LIMIT %s
                        """,
                        (project, limit),
                    )
                elif view == "unused":
                    cur.execute(
                        """
                        SELECT c.content_hash, c.source, c.text, c.token_count,
                               0 AS retrievals, NULL AS avg_score
                        FROM chunks c
                        WHERE c.project = %s AND NOT EXISTS (
                            SELECT 1 FROM chunk_retrievals cr
                            WHERE cr.project = c.project AND cr.content_hash = c.content_hash)
                        LIMIT %s
                        """,
                        (project, limit),
                    )
                elif view == "duplicates":
                    cur.execute(
                        """
                        SELECT c.content_hash,
                               STRING_AGG(DISTINCT c.source, ',') AS source,
                               MIN(c.text) AS text, MIN(c.token_count) AS token_count,
                               COUNT(DISTINCT c.source) AS copies,
                               (SELECT COUNT(*) FROM chunk_retrievals cr
                                 WHERE cr.project=c.project AND cr.content_hash=c.content_hash) AS retrievals
                        FROM chunks c WHERE c.project = %s
                        GROUP BY c.content_hash HAVING COUNT(DISTINCT c.source) > 1
                        ORDER BY copies DESC LIMIT %s
                        """,
                        (project, limit),
                    )
                else:
                    raise ValueError(f"unknown chunk view: {view}")
                return [dict(r) for r in cur.fetchall()]

    def cost_summary(self, project: str) -> Dict[str, Any]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT trace_id, name, start_time,
                           attributes->>'model'                        AS model,
                           attributes->>'provider'                     AS provider,
                           (attributes->>'cost')::numeric              AS cost,
                           (attributes->>'input_tokens')::bigint       AS input_tokens,
                           (attributes->>'output_tokens')::bigint      AS output_tokens,
                           attributes->>'replayed'                     AS replayed,
                           duration_ms
                    FROM events WHERE project=%s AND stage='generation'
                    ORDER BY start_time
                    """,
                    (project,),
                )
                gens = [dict(r) for r in cur.fetchall()]

        total_cost = sum(float(g["cost"] or 0) for g in gens)
        total_in = sum(int(g["input_tokens"] or 0) for g in gens)
        total_out = sum(int(g["output_tokens"] or 0) for g in gens)

        by_model: Dict[str, Dict[str, Any]] = {}
        by_day: Dict[str, Dict[str, Any]] = {}
        for g in gens:
            model = g["model"] or "unknown"
            m = by_model.setdefault(model, {"model": model, "calls": 0, "cost": 0.0,
                                            "input_tokens": 0, "output_tokens": 0,
                                            "latency_ms": 0.0, "_lat_n": 0})
            m["calls"] += 1
            m["cost"] += float(g["cost"] or 0)
            m["input_tokens"] += int(g["input_tokens"] or 0)
            m["output_tokens"] += int(g["output_tokens"] or 0)
            if g["duration_ms"]:
                m["latency_ms"] += float(g["duration_ms"])
                m["_lat_n"] += 1

            day = time.strftime("%Y-%m-%d", time.localtime(float(g["start_time"] or 0)))
            d = by_day.setdefault(day, {"day": day, "calls": 0, "cost": 0.0,
                                        "input_tokens": 0, "output_tokens": 0})
            d["calls"] += 1
            d["cost"] += float(g["cost"] or 0)
            d["input_tokens"] += int(g["input_tokens"] or 0)
            d["output_tokens"] += int(g["output_tokens"] or 0)

        models = []
        for m in by_model.values():
            m["avg_latency_ms"] = (m.pop("latency_ms") / m["_lat_n"]) if m["_lat_n"] else None
            m.pop("_lat_n")
            models.append(m)
        models.sort(key=lambda x: x["cost"], reverse=True)

        return {
            "totals": {
                "generations": len(gens), "cost": total_cost,
                "input_tokens": total_in, "output_tokens": total_out,
                "total_tokens": total_in + total_out, "models": len(by_model),
            },
            "by_model": models,
            "by_day": sorted(by_day.values(), key=lambda x: x["day"]),
        }

    def list_generations(self, project: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_id, trace_id, name, start_time, duration_ms, status,
                           attributes->>'model'                   AS model,
                           attributes->>'provider'                AS provider,
                           (attributes->>'cost')::numeric         AS cost,
                           (attributes->>'input_tokens')::bigint  AS input_tokens,
                           (attributes->>'output_tokens')::bigint AS output_tokens,
                           attributes->>'replayed'                AS replayed,
                           attributes->>'response'                AS response,
                           attributes->>'prompt'                  AS prompt
                    FROM events WHERE project=%s AND stage='generation'
                    ORDER BY start_time DESC LIMIT %s
                    """,
                    (project, limit),
                )
                rows = cur.fetchall()
                out = []
                for r in rows:
                    d = dict(r)
                    cur.execute("SELECT query FROM traces WHERE trace_id=%s", (d["trace_id"],))
                    tq = cur.fetchone()
                    d["query"] = tq["query"] if tq else None
                    out.append(d)
        return out

    def get_generation_context(self, trace_id: str) -> Optional[Dict[str, Any]]:
        t = self.get_trace(trace_id)
        if t is None:
            return None
        ctx = {"trace_id": trace_id, "query": t["trace"].get("query"),
               "system_prompt": None, "final_prompt": None, "model": None, "chunks": []}
        for ev in t["events"]:
            a = ev.get("attributes") or {}
            if ev["stage"] == "context_assembly":
                ctx["final_prompt"] = a.get("final_prompt") or ctx["final_prompt"]
                ctx["system_prompt"] = a.get("system_prompt") or ctx["system_prompt"]
                ctx["chunks"] = a.get("chunks") or ctx["chunks"]
            elif ev["stage"] == "generation":
                ctx["final_prompt"] = ctx["final_prompt"] or a.get("prompt")
                ctx["model"] = a.get("model") or ctx["model"]
        return ctx

    def traces_with_ground_truth(self, project: str) -> List[Dict[str, Any]]:
        from .events import content_hash

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT trace_id, relevant_chunk_ids FROM ground_truth WHERE project=%s",
                    (project,),
                )
                gts = cur.fetchall()
                out = []
                for gt in gts:
                    cur.execute(
                        "SELECT stage, attributes FROM events WHERE trace_id=%s"
                        " AND stage IN ('reranking','retrieval')"
                        " ORDER BY CASE stage WHEN 'reranking' THEN 0 ELSE 1 END, start_time DESC LIMIT 1",
                        (gt["trace_id"],),
                    )
                    ev = cur.fetchone()
                    if ev is None:
                        continue
                    attrs = ev["attributes"] or {}
                    results = attrs.get("after") if ev["stage"] == "reranking" else attrs.get("results")
                    ranked = []
                    for item in results or []:
                        if isinstance(item, dict):
                            cid = item.get("id") or (content_hash(item["text"]) if item.get("text") else None)
                            if cid:
                                ranked.append(cid)
                    out.append({
                        "trace_id": gt["trace_id"],
                        "ranked": ranked,
                        "relevant": gt["relevant_chunk_ids"] or [],
                    })
        return out

    # ---------------------------------------------------------------- eval scores

    def set_eval_score(
        self, trace_id: str, project: str, metric: str,
        score: Optional[float], reason: str = "", model: str = "",
    ) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO eval_scores(trace_id, project, metric, score, reason, model, evaluated_at)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s)"
                    " ON CONFLICT (trace_id, metric) DO UPDATE SET"
                    "   score=EXCLUDED.score, reason=EXCLUDED.reason,"
                    "   model=EXCLUDED.model, evaluated_at=EXCLUDED.evaluated_at",
                    (trace_id, project, metric, score, reason, model, time.time()),
                )

    def get_eval_scores(self, trace_id: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT metric, score, reason, model, evaluated_at FROM eval_scores WHERE trace_id=%s",
                    (trace_id,),
                )
                return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# MultiStore — fan-out writes, primary-backend reads
# ---------------------------------------------------------------------------

class MultiStore:
    """Fan-out writes to multiple backends; reads delegate to the first backend
    that supports them (the primary, i.e. the first in the list).

    Typical use — durable archive + local dashboard::

        store = ragobserve.MultiStore([
            ragobserve.SQLiteStore(),                   # primary: dashboard reads
            ragobserve.FileStore("s3://my-bucket/"),    # sink: durable archive
        ])
        ragobserve.init(project="prod", store=store)
    """

    def __init__(self, backends: List[Any]):
        if not backends:
            raise ValueError("MultiStore requires at least one backend")
        self._backends = backends

    def ingest_events(self, events: List[Dict[str, Any]]) -> int:
        n = 0
        for b in self._backends:
            try:
                n = b.ingest_events(events)
            except Exception as _e:
                _log.warning("MultiStore backend %s.ingest_events failed: %s", type(b).__name__, _e)
        return n

    def set_ground_truth(self, trace_id: str, project: str, relevant_chunk_ids: List[str]) -> None:
        for b in self._backends:
            try:
                b.set_ground_truth(trace_id, project, relevant_chunk_ids)
            except Exception as _e:
                _log.warning("MultiStore backend %s.set_ground_truth failed: %s", type(b).__name__, _e)

    def close(self) -> None:
        for b in self._backends:
            try:
                b.close()
            except Exception:
                pass

    def __getattr__(self, name: str) -> Any:
        """Delegate read methods to the first backend whose method is not write-only."""
        for b in self._backends:
            method = getattr(b, name, None)
            if method is None:
                continue
            underlying = getattr(method, "__func__", method)
            if getattr(underlying, "_is_not_supported", False):
                continue
            return method
        raise AttributeError(f"No backend in MultiStore supports '{name}'")
