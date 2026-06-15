"""Diagnostics for the framework adapters.

Adapters hook into LangChain / LlamaIndex internals (callback signatures,
instrumentation event names, expected methods). Those move between framework
versions, and when they do the failure is silent — a stage just stops being
captured. These helpers turn that silence into a visible ``RagObserveWarning`` so
version drift is noticed instead of producing empty dashboards.
"""
from __future__ import annotations

import warnings
from typing import Iterable


class RagObserveWarning(UserWarning):
    """Emitted when an adapter can't hook something it expected to."""


def warn(message: str) -> None:
    warnings.warn(f"[ragobserve] {message}", RagObserveWarning, stacklevel=3)


def require_methods(obj: object, methods: Iterable[str], what: str) -> None:
    """Warn if ``obj`` is missing every one of ``methods`` (so the wrapper would
    silently capture nothing). ``methods`` is treated as "at least one must
    exist"."""
    present = [m for m in methods if callable(getattr(obj, m, None))]
    if not present:
        warn(
            f"{what}: {type(obj).__name__} has none of {list(methods)} — "
            f"that stage will not be captured (framework version drift?)"
        )
