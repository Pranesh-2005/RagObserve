"""Model price book and cost estimation (Langfuse-style cost tracing).

Prices are USD per 1,000,000 tokens, ``(input, output)``. They are used to
(a) estimate the cost of a *live* generation replay when the provider reports
token usage, and (b) backfill cost on logged generations that didn't carry an
explicit ``cost`` so the cost dashboards have something to show.

Three layers, highest priority first:

1. ``~/.ragobserve/prices.json`` — refreshed from the community-maintained
   LiteLLM price feed via ``ragobserve prices --refresh``. ~2000 chat models
   across ~80 providers, so no provider is favoured over another.
2. ``PRICE_BOOK`` below — offline fallback. Hand-maintained, so it lags.
3. Unknown model → ``None`` (dashboard shows no cost rather than a wrong one).

This table is intentionally editable — RAGObserve is local-first, so users can
add their own models here without waiting on a release.
"""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from typing import Dict, Optional, Tuple

# Community-maintained, all-provider price feed (MIT). One file, every vendor.
FEED_URL = os.getenv(
    "RAGOBSERVE_PRICE_FEED",
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
)

# model id (lowercased) -> (input $/1M, output $/1M)
# ponytail: offline fallback only — vendors reprice often, so run
# `ragobserve prices --refresh` for current numbers rather than editing this.
PRICE_BOOK: Dict[str, Tuple[float, float]] = {
    # Anthropic
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-opus-4-1": (15.0, 75.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-3-7-sonnet": (3.0, 15.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.80, 4.0),
    "claude-3-haiku": (0.25, 1.25),
    # OpenAI
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o3": (2.0, 8.0),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    "o1": (15.0, 60.0),
    "o1-mini": (1.10, 4.40),
    # Google
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.30),
    # xAI
    "grok-4": (3.0, 15.0),
    "grok-3": (3.0, 15.0),
    "grok-3-mini": (0.30, 0.50),
    "grok-2": (2.0, 10.0),
    # Groq (open-weight models, hosted)
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-70b": (0.59, 0.79),
    "llama3-8b-8192": (0.05, 0.08),
    "llama3-70b-8192": (0.59, 0.79),
    "llama-4-scout": (0.11, 0.34),
    "llama-4-maverick": (0.20, 0.60),
    "mixtral-8x7b-32768": (0.24, 0.24),
    "gemma2-9b-it": (0.20, 0.20),
    "qwen3-32b": (0.29, 0.59),
    "kimi-k2": (1.0, 3.0),
    # Mistral
    "mistral-large": (2.0, 6.0),
    "mistral-medium": (0.40, 2.0),
    "mistral-small": (0.20, 0.60),
    "magistral-medium": (2.0, 5.0),
    "codestral": (0.30, 0.90),
    "open-mistral-nemo": (0.15, 0.15),
    # DeepSeek
    "deepseek-chat": (0.28, 0.42),
    "deepseek-reasoner": (0.55, 2.19),
    # Cohere
    "command-a": (2.5, 10.0),
    "command-r-plus": (2.5, 10.0),
    "command-r": (0.15, 0.60),
    # Amazon
    "amazon.nova-pro": (0.80, 3.20),
    "amazon.nova-lite": (0.06, 0.24),
    "amazon.nova-micro": (0.035, 0.14),
    # Alibaba
    "qwen-max": (1.60, 6.40),
    "qwen-plus": (0.40, 1.20),
    "qwen-turbo": (0.05, 0.20),
    # Local / self-hosted — free to run
    "ollama": (0.0, 0.0),
    "local": (0.0, 0.0),
}

# Refreshed feed, loaded lazily from disk. Same shape as PRICE_BOOK.
_FEED: Optional[Dict[str, Tuple[float, float]]] = None


def cache_path() -> str:
    """Where the refreshed feed is cached."""
    return os.path.join(
        os.getenv("RAGOBSERVE_HOME") or os.path.join(os.path.expanduser("~"), ".ragobserve"),
        "prices.json",
    )


def _parse_feed(raw: Dict) -> Dict[str, Tuple[float, float]]:
    """LiteLLM feed -> {model: (input $/1M, output $/1M)}. Costs are per-token there."""
    out: Dict[str, Tuple[float, float]] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict) or spec.get("mode") != "chat":
            continue
        inp, outp = spec.get("input_cost_per_token"), spec.get("output_cost_per_token")
        if not isinstance(inp, (int, float)) or not isinstance(outp, (int, float)):
            continue
        key = name.lower().strip()
        out[key] = (inp * 1e6, outp * 1e6)
        # feed keys are often "provider/model"; index the bare model name too so
        # a user logging model="llama-3.3-70b-versatile" still resolves.
        bare = key.rsplit("/", 1)[-1]
        out.setdefault(bare, (inp * 1e6, outp * 1e6))
    return out


def refresh(url: Optional[str] = None, timeout: float = 30.0) -> int:
    """Download the price feed and cache it. Returns the number of models stored."""
    import httpx

    raw = httpx.get(url or FEED_URL, timeout=timeout, follow_redirects=True).json()
    prices = _parse_feed(raw)
    if not prices:
        raise ValueError("price feed contained no usable chat models")
    path = cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"updated_at": time.time(), "source": url or FEED_URL, "prices": prices}, fh)
    _reset_cache()
    return len(prices)


def _load_feed() -> Dict[str, Tuple[float, float]]:
    global _FEED
    if _FEED is None:
        try:
            with open(cache_path(), encoding="utf-8") as fh:
                _FEED = {k: tuple(v) for k, v in json.load(fh)["prices"].items()}
        except Exception:
            _FEED = {}  # no refresh yet, or corrupt cache — fall back to PRICE_BOOK
    return _FEED


def _reset_cache() -> None:
    global _FEED
    _FEED = None
    _lookup.cache_clear()


def feed_info() -> Optional[Dict]:
    """Metadata about the cached feed, or None if never refreshed."""
    try:
        with open(cache_path(), encoding="utf-8") as fh:
            meta = json.load(fh)
        return {"updated_at": meta.get("updated_at"), "source": meta.get("source"),
                "count": len(meta.get("prices") or {}), "path": cache_path()}
    except Exception:
        return None


@lru_cache(maxsize=2048)
def _lookup(model: Optional[str]) -> Optional[Tuple[float, float]]:
    if not model:
        return None
    m = model.lower().strip()
    feed = _load_feed()
    if m in feed:
        return feed[m]
    if m in PRICE_BOOK:
        return PRICE_BOOK[m]
    # prefix / substring match so dated snapshots (gpt-4o-2024-..) still resolve.
    # Longest key first, so "gpt-4o-mini" never gets matched by "gpt-4o".
    for book in (feed, PRICE_BOOK):
        best = None
        for key, price in book.items():
            if (m.startswith(key) or key in m) and (best is None or len(key) > len(best[0])):
                best = (key, price)
        if best is not None:
            return best[1]
    return None


def estimate_cost(model: Optional[str], input_tokens: Optional[int],
                  output_tokens: Optional[int]) -> Optional[float]:
    """Return the USD cost for a generation, or ``None`` if the model is unknown."""
    price = _lookup(model)
    if price is None:
        return None
    inp = (input_tokens or 0) / 1_000_000.0 * price[0]
    out = (output_tokens or 0) / 1_000_000.0 * price[1]
    return round(inp + out, 6)


def is_known(model: Optional[str]) -> bool:
    return _lookup(model) is not None
