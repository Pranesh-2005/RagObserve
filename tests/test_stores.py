"""Tests for pluggable store backends: PostgresStore, FileStore, MultiStore.

Postgres tests run against the local DB defined in .env (skipped if absent).
S3 tests run against a mocked AWS environment via moto (skipped if moto absent).
FileStore local tests need no external deps.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load .env without requiring python-dotenv
# ---------------------------------------------------------------------------

def _load_env() -> None:
    env = Path(__file__).parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_env()


def _pg_dsn() -> str:
    from urllib.parse import quote_plus
    return (
        f"postgresql://{os.environ['PG_USER']}:{quote_plus(os.environ['PG_PASS'])}"
        f"@{os.environ['PG_HOST']}:{os.environ['PG_PORT']}/{os.environ['DB_NAME']}"
    )


_has_pg = all(k in os.environ for k in ("PG_USER", "PG_PASS", "PG_HOST", "PG_PORT", "DB_NAME"))
try:
    import fsspec as _fsspec  # noqa: F401
    _has_fsspec = True
except ImportError:
    _has_fsspec = False

try:
    import s3fs as _s3fs  # noqa: F401
    _has_s3fs = True
except ImportError:
    _has_s3fs = False

_s3_bucket = os.environ.get("S3_BUCKET", "")

requires_pg     = pytest.mark.skipif(not _has_pg,     reason="Postgres .env vars not set")
requires_fsspec = pytest.mark.skipif(not _has_fsspec, reason="fsspec not installed")
requires_s3     = pytest.mark.skipif(
    not (_has_s3fs and _s3_bucket),
    reason="s3fs not installed or S3_BUCKET not set",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pg_store():
    from ragobserve.stores import PostgresStore
    store = PostgresStore(_pg_dsn())
    yield store
    conn = store._pool.getconn()
    try:
        with conn.cursor() as cur:
            for tbl in ("chunk_retrievals", "chunks", "ground_truth", "events", "traces", "projects"):
                cur.execute(f"TRUNCATE {tbl} CASCADE")
        conn.commit()
    finally:
        store._pool.putconn(conn)
    store.close()


@pytest.fixture
def local_fs(tmp_path):
    from ragobserve.stores import FileStore
    return FileStore(str(tmp_path / "events"))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ev(stage="retrieval", project="test", **attrs):
    import time
    return {
        "event_id":  uuid.uuid4().hex,
        "trace_id":  uuid.uuid4().hex,
        "span_id":   uuid.uuid4().hex,
        "project":   project,
        "stage":     stage,
        "name":      stage,
        "start_time": time.time(),
        "end_time":   time.time(),
        "duration_ms": 10.0,
        "status":    "ok",
        "attributes": {"query": "q?", **attrs},
    }


# ---------------------------------------------------------------------------
# PostgresStore
# ---------------------------------------------------------------------------

class TestPostgresStore:

    @requires_pg
    def test_tables_auto_created(self, pg_store):
        conn = pg_store._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='public' AND table_name = ANY(%s)",
                    (["projects", "traces", "events", "chunks", "chunk_retrievals", "ground_truth"],),
                )
                found = {r["table_name"] for r in cur.fetchall()}
        finally:
            pg_store._pool.putconn(conn)
        assert found == {"projects", "traces", "events", "chunks", "chunk_retrievals", "ground_truth"}

    @requires_pg
    def test_ingest_and_get_trace(self, pg_store):
        ev = _ev(stage="retrieval", results=[{"text": "hello", "score": 0.9}])
        pg_store.ingest_events([ev])
        data = pg_store.get_trace(ev["trace_id"])
        assert data is not None
        assert data["trace"]["trace_id"] == ev["trace_id"]
        assert data["events"][0]["stage"] == "retrieval"

    @requires_pg
    def test_trace_bounds_extend(self, pg_store):
        tid = uuid.uuid4().hex
        e1 = _ev(stage="retrieval");  e1["trace_id"] = tid; e1["start_time"] = 1000.0; e1["end_time"] = 1001.0
        e2 = _ev(stage="generation"); e2["trace_id"] = tid; e2["start_time"] = 1002.0; e2["end_time"] = 1005.0
        pg_store.ingest_events([e1, e2])
        t = pg_store.get_trace(tid)["trace"]
        assert t["start_time"] == 1000.0
        assert t["end_time"]   == 1005.0

    @requires_pg
    def test_list_traces(self, pg_store):
        pg_store.ingest_events([_ev(project="lp"), _ev(project="lp")])
        assert len(pg_store.list_traces("lp")) == 2

    @requires_pg
    def test_list_projects(self, pg_store):
        pg_store.ingest_events([_ev(project="alpha"), _ev(project="beta")])
        names = {p["name"] for p in pg_store.list_projects()}
        assert {"alpha", "beta"}.issubset(names)

    @requires_pg
    def test_cost_summary(self, pg_store):
        ev = _ev(stage="generation", project="cost")
        ev["attributes"] = {"model": "gpt-4o", "input_tokens": 100,
                            "output_tokens": 50, "cost": 0.005}
        pg_store.ingest_events([ev])
        s = pg_store.cost_summary("cost")
        assert s["totals"]["generations"] == 1
        assert s["totals"]["cost"] == pytest.approx(0.005, abs=1e-6)
        assert s["by_model"][0]["model"] == "gpt-4o"

    @requires_pg
    def test_cost_auto_backfilled(self, pg_store):
        ev = _ev(stage="generation", project="bf")
        ev["attributes"] = {"model": "gpt-4o-mini", "input_tokens": 1000,
                            "output_tokens": 500, "cost": None}
        pg_store.ingest_events([ev])
        rows = pg_store.list_generations("bf")
        assert rows and float(rows[0]["cost"] or 0) > 0

    @requires_pg
    def test_error_status_propagates(self, pg_store):
        ev = _ev(); ev["status"] = "error"
        pg_store.ingest_events([ev])
        assert pg_store.get_trace(ev["trace_id"])["trace"]["status"] == "error"

    @requires_pg
    def test_ground_truth_and_metrics(self, pg_store):
        from ragobserve.events import content_hash
        text = "relevant chunk"
        cid  = content_hash(text)
        ev = _ev(stage="retrieval", results=[{"text": text, "score": 0.9, "id": cid}])
        pg_store.ingest_events([ev])
        pg_store.set_ground_truth(ev["trace_id"], "test", [cid])
        pairs = pg_store.traces_with_ground_truth("test")
        assert len(pairs) == 1
        assert pairs[0]["ranked"][0] == cid
        assert pairs[0]["relevant"]  == [cid]

    @requires_pg
    def test_chunk_views_top(self, pg_store):
        ev = _ev(stage="retrieval",
                 results=[{"text": "a", "score": 0.9}, {"text": "b", "score": 0.7}])
        pg_store.ingest_events([ev])
        chunks = pg_store.chunk_views("test", "top")
        assert len(chunks) == 2

    @requires_pg
    def test_chunk_views_unused(self, pg_store):
        ev = _ev(stage="chunking", chunks=[{"text": "orphan chunk"}])
        pg_store.ingest_events([ev])
        unused = pg_store.chunk_views("test", "unused")
        assert any("orphan" in (r.get("text") or "") for r in unused)

    @requires_pg
    def test_generation_context(self, pg_store):
        tid = uuid.uuid4().hex
        ctx_ev = _ev(stage="context_assembly"); ctx_ev["trace_id"] = tid
        ctx_ev["attributes"] = {"final_prompt": "PROMPT", "system_prompt": "SYS",
                                 "query": "q", "chunks": []}
        gen_ev = _ev(stage="generation"); gen_ev["trace_id"] = tid
        gen_ev["attributes"] = {"model": "gpt-4o", "prompt": "PROMPT", "response": "ANS",
                                 "input_tokens": 100, "output_tokens": 50}
        pg_store.ingest_events([ctx_ev, gen_ev])
        ctx = pg_store.get_generation_context(tid)
        assert ctx["final_prompt"] == "PROMPT"
        assert ctx["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# FileStore — local filesystem
# ---------------------------------------------------------------------------

class TestFileStoreLocal:

    def test_ingest_creates_jsonl(self, local_fs, tmp_path):
        ev = _ev(project="demo")
        local_fs.ingest_events([ev])
        files = list((tmp_path / "events").rglob("*.jsonl"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["event_id"] == ev["event_id"]

    def test_returns_event_count(self, local_fs):
        evs = [_ev() for _ in range(4)]
        assert local_fs.ingest_events(evs) == 4

    def test_directory_layout(self, local_fs, tmp_path):
        import time
        ev = _ev(project="myproj")
        local_fs.ingest_events([ev])
        today = time.strftime("%Y-%m-%d")
        assert (tmp_path / "events" / "myproj" / today).is_dir()

    def test_ground_truth_written(self, local_fs, tmp_path):
        local_fs.set_ground_truth("tid-1", "proj", ["c1", "c2"])
        files = list((tmp_path / "events").rglob("gt_*.jsonl"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["trace_id"] == "tid-1"
        assert data["relevant_chunk_ids"] == ["c1", "c2"]

    def test_read_methods_raise(self, local_fs):
        for name in ("list_projects", "list_traces", "get_trace",
                     "chunk_views", "cost_summary", "list_generations",
                     "get_generation_context", "traces_with_ground_truth"):
            with pytest.raises(NotImplementedError):
                getattr(local_fs, name)()

    def test_close_is_noop(self, local_fs):
        local_fs.close()   # must not raise


# ---------------------------------------------------------------------------
# FileStore — memory:// backend (fsspec built-in; same code path as s3://)
# ---------------------------------------------------------------------------

class TestFileStoreS3:

    @requires_fsspec
    def test_events_land_in_bucket(self):
        from ragobserve.stores import FileStore
        root = f"memory://trag-{uuid.uuid4().hex[:6]}/events"
        store = FileStore(root)
        ev = _ev(project="s3proj")
        n = store.ingest_events([ev])
        assert n == 1
        files = store._fs.find(store._root)
        assert len(files) == 1
        data = json.loads(store._fs.open(files[0], "rb").read())
        assert data["event_id"] == ev["event_id"]

    @requires_fsspec
    def test_multiple_events_multiple_keys(self):
        from ragobserve.stores import FileStore
        root = f"memory://multi-{uuid.uuid4().hex[:6]}"
        store = FileStore(root)
        evs = [_ev(project="p") for _ in range(3)]
        store.ingest_events(evs)
        files = store._fs.find(store._root)
        assert len(files) == 3

    @requires_fsspec
    def test_ground_truth_in_bucket(self):
        from ragobserve.stores import FileStore
        root = f"memory://gt-{uuid.uuid4().hex[:6]}"
        store = FileStore(root)
        store.set_ground_truth("tid-xyz", "proj", ["c1", "c2"])
        files = store._fs.find(store._root)
        gt_files = [f for f in files if "gt_" in f]
        assert len(gt_files) == 1
        data = json.loads(store._fs.open(gt_files[0], "rb").read())
        assert data["relevant_chunk_ids"] == ["c1", "c2"]


# ---------------------------------------------------------------------------
# FileStore — real S3 bucket (ragobs-tst)
# ---------------------------------------------------------------------------

class TestFileStoreS3Integration:
    """Writes to the real S3_BUCKET and verifies via boto3. Skipped if bucket/s3fs absent."""

    @requires_s3
    def test_event_lands_in_real_bucket(self):
        import boto3
        from ragobserve.stores import FileStore

        prefix = f"test-{uuid.uuid4().hex[:8]}"
        bucket = _s3_bucket
        store = FileStore(f"s3://{bucket}/{prefix}/")
        ev = _ev(project="s3int")
        assert store.ingest_events([ev]) == 1

        s3 = boto3.client("s3")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/s3int/")
        keys = [o["Key"] for o in resp.get("Contents", [])]
        assert len(keys) == 1
        body = s3.get_object(Bucket=bucket, Key=keys[0])["Body"].read()
        assert json.loads(body)["event_id"] == ev["event_id"]

        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in keys]})

    @requires_s3
    def test_ground_truth_in_real_bucket(self):
        import boto3
        from ragobserve.stores import FileStore

        prefix = f"test-gt-{uuid.uuid4().hex[:8]}"
        bucket = _s3_bucket
        store = FileStore(f"s3://{bucket}/{prefix}/")
        store.set_ground_truth("tid-s3", "proj", ["c1", "c2"])

        s3 = boto3.client("s3")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/proj/")
        keys = [o["Key"] for o in resp.get("Contents", [])]
        gt_keys = [k for k in keys if "gt_" in k]
        assert len(gt_keys) == 1
        data = json.loads(s3.get_object(Bucket=bucket, Key=gt_keys[0])["Body"].read())
        assert data["relevant_chunk_ids"] == ["c1", "c2"]

        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in gt_keys]})


# ---------------------------------------------------------------------------
# MultiStore
# ---------------------------------------------------------------------------

class TestMultiStore:

    def test_fanout_writes_all_backends(self, tmp_path):
        from ragobserve.stores import FileStore, MultiStore, SQLiteStore
        sql = SQLiteStore(str(tmp_path / "a.db"))
        fs  = FileStore(str(tmp_path / "ev"))
        store = MultiStore([sql, fs])
        ev = _ev(project="multi")
        store.ingest_events([ev])
        assert sql.get_trace(ev["trace_id"]) is not None
        assert list((tmp_path / "ev").rglob("*.jsonl"))

    def test_reads_from_primary(self, tmp_path):
        from ragobserve.stores import FileStore, MultiStore, SQLiteStore
        sql = SQLiteStore(str(tmp_path / "b.db"))
        store = MultiStore([sql, FileStore(str(tmp_path / "ev2"))])
        ev = _ev(project="r")
        store.ingest_events([ev])
        data = store.get_trace(ev["trace_id"])
        assert data is not None
        assert data["trace"]["project"] == "r"

    def test_skips_write_only_backend_for_reads(self, tmp_path):
        from ragobserve.stores import FileStore, MultiStore, SQLiteStore
        # FileStore first (write-only), SQLiteStore second
        fs  = FileStore(str(tmp_path / "ev3"))
        sql = SQLiteStore(str(tmp_path / "c.db"))
        store = MultiStore([fs, sql])
        ev = _ev(project="skip")
        store.ingest_events([ev])
        # must delegate get_trace to SQLiteStore, skipping FileStore
        data = store.get_trace(ev["trace_id"])
        assert data is not None

    def test_secondary_failure_does_not_crash(self, tmp_path):
        from ragobserve.stores import MultiStore, SQLiteStore

        class BrokenStore:
            def ingest_events(self, events): raise RuntimeError("boom")
            def set_ground_truth(self, *a): raise RuntimeError("boom")
            def close(self): pass

        sql = SQLiteStore(str(tmp_path / "d.db"))
        store = MultiStore([sql, BrokenStore()])
        ev = _ev(project="resilience")
        store.ingest_events([ev])   # must not raise
        assert sql.get_trace(ev["trace_id"]) is not None

    def test_ground_truth_fans_out(self, tmp_path):
        from ragobserve.stores import MultiStore, SQLiteStore
        sql1 = SQLiteStore(str(tmp_path / "e1.db"))
        sql2 = SQLiteStore(str(tmp_path / "e2.db"))
        store = MultiStore([sql1, sql2])
        ev = _ev(stage="retrieval", results=[{"text": "x", "score": 0.9}])
        store.ingest_events([ev])
        store.set_ground_truth(ev["trace_id"], "test", ["abc"])
        assert sql1.traces_with_ground_truth("test")
        assert sql2.traces_with_ground_truth("test")

    def test_requires_at_least_one_backend(self):
        from ragobserve.stores import MultiStore
        with pytest.raises(ValueError):
            MultiStore([])

    def test_close_calls_all_backends(self, tmp_path):
        from ragobserve.stores import MultiStore, SQLiteStore
        closed = []

        class TrackingStore:
            def ingest_events(self, e): return 0
            def set_ground_truth(self, *a): pass
            def close(self): closed.append(True)

        sql = SQLiteStore(str(tmp_path / "f.db"))
        store = MultiStore([sql, TrackingStore()])
        store.close()
        assert closed == [True]


# ---------------------------------------------------------------------------
# BaseStore protocol
# ---------------------------------------------------------------------------

def test_protocol_sqlite(tmp_path):
    from ragobserve.stores import BaseStore, SQLiteStore
    assert isinstance(SQLiteStore(str(tmp_path / "p.db")), BaseStore)


@requires_pg
def test_protocol_postgres():
    from ragobserve.stores import BaseStore, PostgresStore
    store = PostgresStore(_pg_dsn())
    assert isinstance(store, BaseStore)
    store.close()


def test_protocol_filestore(tmp_path):
    from ragobserve.stores import BaseStore, FileStore
    assert isinstance(FileStore(str(tmp_path / "ev")), BaseStore)


def test_protocol_multistore(tmp_path):
    from ragobserve.stores import BaseStore, MultiStore, SQLiteStore
    store = MultiStore([SQLiteStore(str(tmp_path / "m.db"))])
    assert isinstance(store, BaseStore)


# ---------------------------------------------------------------------------
# init(store=...) integration
# ---------------------------------------------------------------------------

def test_init_with_custom_store(tmp_path):
    import ragobserve
    from ragobserve import client as rclient
    from ragobserve.stores import SQLiteStore

    store = SQLiteStore(str(tmp_path / "custom.db"))
    ragobserve.init(project="custom-test", store=store)

    captured_tid = None
    with ragobserve.trace("query", query="hello") as span:
        captured_tid = span.trace_id
        ragobserve.log_retrieval("hello", [{"text": "chunk", "score": 0.8}])

    rclient._config.client = None
    rclient._config.project = "default"

    data = store.get_trace(captured_tid)
    assert data is not None
    stages = [e["stage"] for e in data["events"]]
    assert "retrieval" in stages
