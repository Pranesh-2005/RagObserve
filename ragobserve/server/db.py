"""SQLite storage for RAGObserve.

Local-first by design (MLflow convention): defaults to ``./ragobserve.db`` in the
working directory. Uses stdlib sqlite3 with a process-wide lock; fine for the
local single-user tool this v1 is.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

_log = logging.getLogger("ragobserve")

from ..events import content_hash, estimate_tokens
from . import pricing

MIGRATIONS = [
    (1, """
CREATE TABLE IF NOT EXISTS eval_scores (
    trace_id     TEXT NOT NULL,
    project      TEXT NOT NULL,
    metric       TEXT NOT NULL,
    score        REAL,
    reason       TEXT,
    model        TEXT,
    evaluated_at REAL,
    PRIMARY KEY (trace_id, metric)
);
"""),
]


def _apply_migrations(conn: sqlite3.Connection, lock: threading.Lock) -> None:
    with lock:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_versions "
            "(version INTEGER PRIMARY KEY, applied_at REAL)"
        )
        row = conn.execute("SELECT MAX(version) AS v FROM schema_versions").fetchone()
        current = row["v"] or 0
        for version, sql in MIGRATIONS:
            if version > current:
                conn.executescript(sql.strip())
                conn.execute(
                    "INSERT OR IGNORE INTO schema_versions(version, applied_at) VALUES (?,?)",
                    (version, time.time()),
                )
        conn.commit()


SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    name TEXT PRIMARY KEY,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    name TEXT,
    query TEXT,
    start_time REAL,
    end_time REAL,
    status TEXT DEFAULT 'ok'
);
CREATE INDEX IF NOT EXISTS idx_traces_project ON traces(project);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    span_id TEXT,
    parent_span_id TEXT,
    project TEXT NOT NULL,
    stage TEXT NOT NULL,
    name TEXT,
    start_time REAL,
    end_time REAL,
    duration_ms REAL,
    status TEXT DEFAULT 'ok',
    attributes TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_trace ON events(trace_id);
CREATE INDEX IF NOT EXISTS idx_events_project_stage ON events(project, stage);
CREATE TABLE IF NOT EXISTS chunks (
    project TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    text TEXT,
    token_count INTEGER,
    metadata TEXT DEFAULT '{}',
    first_seen REAL,
    PRIMARY KEY (project, content_hash, source)
);
CREATE TABLE IF NOT EXISTS chunk_retrievals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    project TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    score REAL,
    rank INTEGER
);
CREATE INDEX IF NOT EXISTS idx_cr_project ON chunk_retrievals(project, content_hash);
CREATE INDEX IF NOT EXISTS idx_cr_trace ON chunk_retrievals(trace_id);
CREATE TABLE IF NOT EXISTS ground_truth (
    trace_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    relevant_chunk_ids TEXT NOT NULL
);
"""


def _loads(s: Optional[str]) -> Any:
    try:
        return json.loads(s) if s else {}
    except (TypeError, ValueError):
        return {}


class Store:
    def __init__(self, path: str = "ragobserve.db"):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        _apply_migrations(self._conn, self._lock)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------- ingest

    def ingest_events(self, events: List[Dict[str, Any]]) -> int:
        with self._lock:
            cur = self._conn.cursor()
            for ev in events:
                self._ingest_one(cur, ev)
            self._conn.commit()
        try:
            from . import bus
            for ev in events:
                bus.publish(ev)
        except Exception as _e:
            _log.warning("bus.publish failed: %s", _e)
        return len(events)

    def _ingest_one(self, cur: sqlite3.Cursor, ev: Dict[str, Any]) -> None:
        project = ev.get("project") or "default"
        trace_id = ev.get("trace_id") or ev.get("event_id")
        attrs = ev.get("attributes") or {}
        # backfill generation cost from the price book when not supplied, so the
        # cost dashboards work even if the user didn't pass cost= explicitly.
        if ev.get("stage") == "generation" and attrs.get("cost") in (None, 0):
            est = pricing.estimate_cost(attrs.get("model"), attrs.get("input_tokens"),
                                        attrs.get("output_tokens"))
            if est is not None:
                attrs = {**attrs, "cost": est, "cost_estimated": True}
        cur.execute(
            "INSERT OR IGNORE INTO projects(name, created_at) VALUES (?, ?)",
            (project, time.time()),
        )
        # upsert trace row: extend bounds, capture query from any event carrying one
        cur.execute("SELECT trace_id, query, start_time, end_time, status FROM traces WHERE trace_id=?", (trace_id,))
        row = cur.fetchone()
        start = ev.get("start_time") or time.time()
        end = ev.get("end_time") or start
        query = attrs.get("query")
        name = ev.get("name") or ev.get("stage")
        if row is None:
            cur.execute(
                "INSERT INTO traces(trace_id, project, name, query, start_time, end_time, status) VALUES (?,?,?,?,?,?,?)",
                (trace_id, project, name, query, start, end, ev.get("status", "ok")),
            )
        else:
            new_start = min(row["start_time"] or start, start)
            new_end = max(row["end_time"] or end, end)
            new_query = row["query"] or query
            status = "error" if (row["status"] == "error" or ev.get("status") == "error") else "ok"
            cur.execute(
                "UPDATE traces SET start_time=?, end_time=?, query=?, status=? WHERE trace_id=?",
                (new_start, new_end, new_query, status, trace_id),
            )
        cur.execute(
            "INSERT OR REPLACE INTO events(event_id, trace_id, span_id, parent_span_id, project, stage, name,"
            " start_time, end_time, duration_ms, status, attributes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ev.get("event_id"),
                trace_id,
                ev.get("span_id"),
                ev.get("parent_span_id"),
                project,
                ev.get("stage", "other"),
                ev.get("name", ""),
                ev.get("start_time"),
                ev.get("end_time"),
                ev.get("duration_ms"),
                ev.get("status", "ok"),
                json.dumps(attrs, default=str),
            ),
        )
        self._extract_chunks(cur, project, trace_id, ev.get("event_id"), ev.get("stage"), attrs)

    def _extract_chunks(self, cur, project: str, trace_id: str, event_id: str, stage: str, attrs: Dict) -> None:
        """Register chunks and retrieval hits so chunk analytics work across traces."""
        def register(item: Dict[str, Any]) -> Optional[str]:
            text = item.get("text") or ""
            chash = item.get("id") or (content_hash(text) if text else None)
            if chash is None:
                return None
            cur.execute(
                "INSERT OR IGNORE INTO chunks(project, content_hash, source, text, token_count, metadata, first_seen)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    project,
                    chash,
                    item.get("source") or "",
                    text,
                    item.get("token_count") or (estimate_tokens(text) if text else None),
                    json.dumps(item.get("metadata") or {}, default=str),
                    time.time(),
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
                        " VALUES (?,?,?,?,?,?)",
                        (event_id, trace_id, project, chash, item.get("score"), item.get("rank", i + 1)),
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
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO ground_truth(trace_id, project, relevant_chunk_ids) VALUES (?,?,?)",
                (trace_id, project, json.dumps(relevant_chunk_ids)),
            )
            self._conn.commit()

    # ------------------------------------------------------------- queries

    def list_projects(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT p.name,
                       COUNT(t.trace_id) AS traces,
                       AVG((t.end_time - t.start_time) * 1000.0) AS avg_latency_ms,
                       MAX(t.end_time) AS last_activity
                FROM projects p LEFT JOIN traces t ON t.project = p.name
                GROUP BY p.name ORDER BY last_activity DESC
                """
            ).fetchall()
            out = []
            for r in rows:
                cost = self._conn.execute(
                    "SELECT SUM(json_extract(attributes, '$.cost')) AS c FROM events"
                    " WHERE project=? AND stage IN ('generation','embedding')",
                    (r["name"],),
                ).fetchone()
                out.append(
                    {
                        "name": r["name"],
                        "traces": r["traces"],
                        "avg_latency_ms": r["avg_latency_ms"],
                        "total_cost": cost["c"] or 0,
                        "last_activity": r["last_activity"],
                    }
                )
            return out

    def list_traces(self, project: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        where, args = ("WHERE t.project = ?", [project]) if project else ("", [])
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT t.*,
                  (SELECT SUM(json_extract(e.attributes,'$.cost')) FROM events e
                     WHERE e.trace_id = t.trace_id AND e.stage='generation') AS cost,
                  (SELECT json_extract(e.attributes,'$.model') FROM events e
                     WHERE e.trace_id = t.trace_id AND e.stage='generation' LIMIT 1) AS model,
                  (SELECT json_extract(e.attributes,'$.retriever') FROM events e
                     WHERE e.trace_id = t.trace_id AND e.stage='retrieval' LIMIT 1) AS retriever,
                  (SELECT COUNT(*) FROM chunk_retrievals c WHERE c.trace_id = t.trace_id) AS chunk_count,
                  (SELECT score FROM eval_scores WHERE trace_id=t.trace_id AND metric='faithfulness' LIMIT 1) AS faithfulness_score,
                  (SELECT score FROM eval_scores WHERE trace_id=t.trace_id AND metric='answer_relevance' LIMIT 1) AS answer_relevance_score
                FROM traces t {where}
                ORDER BY t.start_time DESC LIMIT ?
                """,
                args + [limit],
            ).fetchall()
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
        with self._lock:
            t = self._conn.execute("SELECT * FROM traces WHERE trace_id=?", (trace_id,)).fetchone()
            if t is None:
                return None
            evs = self._conn.execute(
                "SELECT * FROM events WHERE trace_id=? ORDER BY start_time", (trace_id,)
            ).fetchall()
            gt = self._conn.execute("SELECT * FROM ground_truth WHERE trace_id=?", (trace_id,)).fetchone()
        events = []
        for e in evs:
            d = dict(e)
            d["attributes"] = _loads(e["attributes"])
            events.append(d)
        trace = dict(t)
        if t["end_time"] and t["start_time"]:
            trace["duration_ms"] = (t["end_time"] - t["start_time"]) * 1000.0
        return {
            "trace": trace,
            "events": events,
            "ground_truth": _loads(gt["relevant_chunk_ids"]) if gt else None,
        }

    def chunk_views(self, project: str, view: str = "top", limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            if view == "top":
                rows = self._conn.execute(
                    """
                    SELECT c.content_hash, MIN(c.source) AS source, MIN(c.text) AS text,
                           MIN(c.token_count) AS token_count,
                           COUNT(cr.id) AS retrievals, AVG(cr.score) AS avg_score
                    FROM chunks c JOIN chunk_retrievals cr
                      ON cr.project = c.project AND cr.content_hash = c.content_hash
                    WHERE c.project = ?
                    GROUP BY c.content_hash ORDER BY retrievals DESC LIMIT ?
                    """,
                    (project, limit),
                ).fetchall()
            elif view == "unused":
                rows = self._conn.execute(
                    """
                    SELECT c.content_hash, c.source, c.text, c.token_count,
                           0 AS retrievals, NULL AS avg_score
                    FROM chunks c
                    WHERE c.project = ? AND NOT EXISTS (
                        SELECT 1 FROM chunk_retrievals cr
                        WHERE cr.project = c.project AND cr.content_hash = c.content_hash)
                    LIMIT ?
                    """,
                    (project, limit),
                ).fetchall()
            elif view == "duplicates":
                rows = self._conn.execute(
                    """
                    SELECT c.content_hash, GROUP_CONCAT(DISTINCT c.source) AS source,
                           MIN(c.text) AS text, MIN(c.token_count) AS token_count,
                           COUNT(DISTINCT c.source) AS copies,
                           (SELECT COUNT(*) FROM chunk_retrievals cr
                             WHERE cr.project=c.project AND cr.content_hash=c.content_hash) AS retrievals
                    FROM chunks c WHERE c.project = ?
                    GROUP BY c.content_hash HAVING copies > 1
                    ORDER BY copies DESC LIMIT ?
                    """,
                    (project, limit),
                ).fetchall()
            else:
                raise ValueError(f"unknown chunk view: {view}")
        return [dict(r) for r in rows]

    # ----------------------------------------------------- cost / generations

    def cost_summary(self, project: str) -> Dict[str, Any]:
        """Langfuse-style cost tracing: totals, per-model breakdown, daily series,
        and token usage — computed from logged ``generation`` events."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT trace_id, name, start_time,
                       json_extract(attributes,'$.model')         AS model,
                       json_extract(attributes,'$.provider')      AS provider,
                       json_extract(attributes,'$.cost')          AS cost,
                       json_extract(attributes,'$.input_tokens')  AS input_tokens,
                       json_extract(attributes,'$.output_tokens') AS output_tokens,
                       json_extract(attributes,'$.replayed')      AS replayed,
                       duration_ms
                FROM events WHERE project=? AND stage='generation'
                ORDER BY start_time
                """,
                (project,),
            ).fetchall()

        gens = [dict(r) for r in rows]
        total_cost = sum((g["cost"] or 0) for g in gens)
        total_in = sum((g["input_tokens"] or 0) for g in gens)
        total_out = sum((g["output_tokens"] or 0) for g in gens)

        by_model: Dict[str, Dict[str, Any]] = {}
        by_day: Dict[str, Dict[str, Any]] = {}
        for g in gens:
            model = g["model"] or "unknown"
            m = by_model.setdefault(model, {"model": model, "calls": 0, "cost": 0.0,
                                            "input_tokens": 0, "output_tokens": 0,
                                            "latency_ms": 0.0, "_lat_n": 0})
            m["calls"] += 1
            m["cost"] += g["cost"] or 0
            m["input_tokens"] += g["input_tokens"] or 0
            m["output_tokens"] += g["output_tokens"] or 0
            if g["duration_ms"]:
                m["latency_ms"] += g["duration_ms"]
                m["_lat_n"] += 1

            day = time.strftime("%Y-%m-%d", time.localtime(g["start_time"] or 0))
            d = by_day.setdefault(day, {"day": day, "calls": 0, "cost": 0.0,
                                        "input_tokens": 0, "output_tokens": 0})
            d["calls"] += 1
            d["cost"] += g["cost"] or 0
            d["input_tokens"] += g["input_tokens"] or 0
            d["output_tokens"] += g["output_tokens"] or 0

        models = []
        for m in by_model.values():
            m["avg_latency_ms"] = (m.pop("latency_ms") / m["_lat_n"]) if m["_lat_n"] else None
            m.pop("_lat_n")
            models.append(m)
        models.sort(key=lambda x: x["cost"], reverse=True)
        days = sorted(by_day.values(), key=lambda x: x["day"])

        return {
            "totals": {
                "generations": len(gens),
                "cost": total_cost,
                "input_tokens": total_in,
                "output_tokens": total_out,
                "total_tokens": total_in + total_out,
                "models": len(by_model),
            },
            "by_model": models,
            "by_day": days,
        }

    def list_generations(self, project: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Generations with the context that produced them, for the generation viewer."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT event_id, trace_id, name, start_time, duration_ms, status,
                       json_extract(attributes,'$.model')         AS model,
                       json_extract(attributes,'$.provider')      AS provider,
                       json_extract(attributes,'$.cost')          AS cost,
                       json_extract(attributes,'$.input_tokens')  AS input_tokens,
                       json_extract(attributes,'$.output_tokens') AS output_tokens,
                       json_extract(attributes,'$.replayed')      AS replayed,
                       attributes
                FROM events WHERE project=? AND stage='generation'
                ORDER BY start_time DESC LIMIT ?
                """,
                (project, limit),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            attrs = _loads(d.pop("attributes"))
            d["response"] = attrs.get("response")
            d["prompt"] = attrs.get("prompt")
            # query carried on the trace
            tq = self._conn.execute("SELECT query FROM traces WHERE trace_id=?", (d["trace_id"],)).fetchone()
            d["query"] = tq["query"] if tq else None
            out.append(d)
        return out

    def get_generation_context(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Pull the prompt/system/context a trace used, so a generation can be
        replayed against the exact same inputs."""
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

    # ----------------------------------------------------- eval scores

    def set_eval_score(
        self, trace_id: str, project: str, metric: str,
        score: Optional[float], reason: str = "", model: str = "",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO eval_scores"
                "(trace_id, project, metric, score, reason, model, evaluated_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (trace_id, project, metric, score, reason, model, time.time()),
            )
            self._conn.commit()

    def get_eval_scores(self, trace_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT metric, score, reason, model, evaluated_at FROM eval_scores WHERE trace_id=?",
                (trace_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def traces_with_ground_truth(self, project: str) -> List[Dict[str, Any]]:
        """Per-trace (final ranked chunk ids, relevant ids) pairs for the eval metrics."""
        with self._lock:
            gts = self._conn.execute(
                "SELECT trace_id, relevant_chunk_ids FROM ground_truth WHERE project=?", (project,)
            ).fetchall()
            out = []
            for gt in gts:
                # prefer reranked order, else retrieval order
                ev = self._conn.execute(
                    "SELECT stage, attributes FROM events WHERE trace_id=? AND stage IN ('reranking','retrieval')"
                    " ORDER BY CASE stage WHEN 'reranking' THEN 0 ELSE 1 END, start_time DESC LIMIT 1",
                    (gt["trace_id"],),
                ).fetchone()
                if ev is None:
                    continue
                attrs = _loads(ev["attributes"])
                results = attrs.get("after") if ev["stage"] == "reranking" else attrs.get("results")
                ranked = []
                for item in results or []:
                    if isinstance(item, dict):
                        cid = item.get("id") or (content_hash(item["text"]) if item.get("text") else None)
                        if cid:
                            ranked.append(cid)
                out.append(
                    {
                        "trace_id": gt["trace_id"],
                        "ranked": ranked,
                        "relevant": _loads(gt["relevant_chunk_ids"]) or [],
                    }
                )
            return out
