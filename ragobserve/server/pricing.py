"""Model price book and cost estimation (Langfuse-style cost tracing).

Prices are USD per 1,000,000 tokens, ``(input, output)``. They are used to
(a) estimate the cost of a *live* generation replay when the provider reports
token usage, and (b) backfill cost on logged generations that didn't carry an
explicit ``cost`` so the cost dashboards have something to show.

This table is intentionally editable — RAGObserve is local-first, so users can
add their own models here without waiting on a release.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

# model id (lowercased) -> (input $/1M, output $/1M)
PRICE_BOOK: Dict[str, Tuple[float, float]] = {
    # Anthropic
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-3-5-haiku": (0.80, 4.0),
    # OpenAI
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3": (2.0, 8.0),
    "o4-mini": (1.10, 4.40),
    # Google
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    # Groq (open-weight models, hosted)
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-70b": (0.59, 0.79),
    "llama3-8b-8192": (0.05, 0.08),
    "llama3-70b-8192": (0.59, 0.79),
    "mixtral-8x7b-32768": (0.24, 0.24),
    "gemma2-9b-it": (0.20, 0.20),
    # Mistral
    "mistral-large": (2.0, 6.0),
    "mistral-small": (0.20, 0.60),
    "open-mistral-nemo": (0.15, 0.15),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # Local / self-hosted — free to run
    "ollama": (0.0, 0.0),
    "local": (0.0, 0.0),
}


def _lookup(model: Optional[str]) -> Optional[Tuple[float, float]]:
    if not model:
        return None
    m = model.lower().strip()
    if m in PRICE_BOOK:
        return PRICE_BOOK[m]
    # prefix / substring match so dated snapshots (gpt-4o-2024-..) still resolve
    for key, price in PRICE_BOOK.items():
        if m.startswith(key) or key in m:
            return price
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
