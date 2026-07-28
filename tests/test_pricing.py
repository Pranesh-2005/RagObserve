"""Price book lookup + refresh. Guards the three bugs fixed in 0.6.0."""
import asyncio
import json
import os
import time

import pytest

from ragobserve.server import pricing


@pytest.fixture(autouse=True)
def _isolate_feed(tmp_path, monkeypatch):
    """Keep every test off the user's real ~/.ragobserve/prices.json."""
    monkeypatch.setenv("RAGOBSERVE_HOME", str(tmp_path))
    pricing._reset_cache()
    yield
    pricing._reset_cache()


def test_longest_match_wins():
    # "gpt-4o" is a substring of "gpt-4o-mini"; first-match order used to return
    # the 16x-more-expensive gpt-4o rate for every mini call.
    assert pricing._lookup("gpt-4o-mini") == (0.15, 0.60)
    assert pricing._lookup("gpt-4o") == (2.5, 10.0)


def test_dated_snapshot_resolves_by_prefix():
    assert pricing._lookup("gpt-4o-2024-08-06") == (2.5, 10.0)


def test_unknown_model_has_no_cost():
    assert pricing._lookup("totally-made-up-model") is None
    assert pricing.estimate_cost("totally-made-up-model", 1000, 100) is None


def test_estimate_cost_math():
    # 1M in + 1M out on claude-opus-4-5 == 5 + 25
    assert pricing.estimate_cost("claude-opus-4-5", 1_000_000, 1_000_000) == pytest.approx(30.0)
    assert pricing.estimate_cost("claude-opus-4-5", None, None) == 0.0


def test_all_providers_represented():
    """No single vendor is the only one priced offline."""
    for probe in ["claude-opus-4-5", "gpt-4o", "gemini-2.5-flash", "grok-4",
                  "deepseek-chat", "mistral-small", "command-r", "llama3-8b-8192",
                  "qwen-max", "amazon.nova-pro"]:
        assert pricing._lookup(probe) is not None, probe


def test_feed_parse_converts_per_token_to_per_million():
    raw = {
        "openai/some-model": {"mode": "chat", "input_cost_per_token": 3e-06,
                              "output_cost_per_token": 1.5e-05},
        "an-embedding": {"mode": "embedding", "input_cost_per_token": 1e-07},
        "no-price": {"mode": "chat"},
    }
    out = pricing._parse_feed(raw)
    assert out["openai/some-model"] == (3.0, 15.0)
    assert out["some-model"] == (3.0, 15.0)  # bare name also indexed
    assert "an-embedding" not in out and "no-price" not in out


def test_feed_overrides_builtin(tmp_path):
    path = pricing.cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"updated_at": time.time(), "source": "test",
                   "prices": {"gpt-4o": [99.0, 999.0]}}, fh)
    pricing._reset_cache()
    assert pricing._lookup("gpt-4o") == (99.0, 999.0)
    # models absent from the feed still fall back to the builtin book
    assert pricing._lookup("claude-opus-4-5") == (5.0, 25.0)


def test_corrupt_feed_falls_back_silently():
    path = pricing.cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not json")
    pricing._reset_cache()
    assert pricing._lookup("gpt-4o") == (2.5, 10.0)
    assert pricing.feed_info() is None


def test_refresh_rejects_empty_feed(monkeypatch):
    class _Resp:
        def json(self):
            return {"only-embeddings": {"mode": "embedding"}}

    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp())
    with pytest.raises(ValueError):
        pricing.refresh()


def test_refresh_writes_cache(monkeypatch):
    class _Resp:
        def json(self):
            return {"vendor/m1": {"mode": "chat", "input_cost_per_token": 1e-06,
                                  "output_cost_per_token": 2e-06}}

    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp())
    n = pricing.refresh()
    assert n >= 1
    info = pricing.feed_info()
    assert info["count"] == n
    assert pricing._lookup("vendor/m1") == (1.0, 2.0)


# ---------------------------------------------------------------- hot path

def test_async_trace_context_manager(tmp_path):
    """`async with ragobserve.trace(...)` is documented in the README; it used to
    raise TypeError because __aenter__ was missing."""
    import ragobserve

    ragobserve.init(project="t", db_path=str(tmp_path / "a.db"))

    async def run():
        async with ragobserve.trace("query", query="q") as span:
            assert span.trace_id
            return span.trace_id

    assert asyncio.run(run())


def test_sqlite_uses_wal(tmp_path):
    """journal_mode=delete cost one fsync per logged event (~96ms on Windows)."""
    from ragobserve.server.db import Store

    s = Store(str(tmp_path / "w.db"))
    assert s._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert s._conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    s.close()
