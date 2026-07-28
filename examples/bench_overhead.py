"""Measure the overhead RAGObserve adds to the instrumented app's hot path.

    python examples/bench_overhead.py

Reports median/p95 for each logger, for a full 4-span trace in both sync and
async form, and for the price-book lookup that runs on the ingest path.
"""
from __future__ import annotations

import asyncio
import os
import statistics
import tempfile
import time

import ragobserve
from ragobserve.events import estimate_tokens
from ragobserve.server import pricing

RESULTS = [
    {"id": f"c{i}", "text": "lorem ipsum dolor sit amet " * 20, "score": 0.9 - i * 0.01}
    for i in range(10)
]
PROMPT = "context:\n" + "\n".join(r["text"] for r in RESULTS) + "\nquestion: what?"


def bench(fn, n: int = 200):
    fn()  # warm up
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return statistics.median(samples), samples[int(n * 0.95)]


def report(label: str, fn, n: int = 200, scale: float = 1.0, unit: str = "ms") -> None:
    med, p95 = bench(fn, n)
    print(f"  {label:30s} median {med * scale:7.3f} {unit}   p95 {p95 * scale:7.3f} {unit}")


def full_trace() -> None:
    with ragobserve.trace("query", query="q"):
        ragobserve.log_retrieval("q", RESULTS, retriever="qdrant")
        ragobserve.log_context(PROMPT, chunks=RESULTS)
        ragobserve.log_generation(model="gpt-4o", prompt=PROMPT, response="hi",
                                  input_tokens=1200, output_tokens=40)


async def async_full_trace() -> None:
    async with ragobserve.trace("query", query="q"):
        ragobserve.log_retrieval("q", RESULTS, retriever="qdrant")
        ragobserve.log_context(PROMPT, chunks=RESULTS)
        ragobserve.log_generation(model="gpt-4o", prompt=PROMPT, response="hi",
                                  input_tokens=1200, output_tokens=40)


async def bench_async(n: int = 100) -> None:
    await async_full_trace()
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        await async_full_trace()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    print(f"  {'full 4-span trace':30s} median {statistics.median(samples):7.3f} ms"
          f"   p95 {samples[int(n * 0.95)]:7.3f} ms")


def main() -> None:
    ragobserve.init(project="bench", db_path=os.path.join(tempfile.mkdtemp(), "bench.db"))

    print("=== sync (LocalClient writes inline) ===")
    report("log_retrieval (10 chunks)",
           lambda: ragobserve.log_retrieval("q", RESULTS, retriever="qdrant"))
    report("log_context (~5KB prompt)",
           lambda: ragobserve.log_context(PROMPT, chunks=RESULTS))
    report("log_generation",
           lambda: ragobserve.log_generation(model="gpt-4o", prompt=PROMPT, response="hi",
                                             input_tokens=1200, output_tokens=40))
    report("full 4-span trace", full_trace, n=100)

    print("\n=== async (write offloaded to executor) ===")
    asyncio.run(bench_async())

    print("\n=== price lookup (server ingest path) ===")
    for model in ["gpt-4o", "gpt-4o-2024-08-06", "unknown-model-v3"]:
        report(model, lambda m=model: pricing.estimate_cost(m, 1200, 40),
               n=5000, scale=1000, unit="us")

    print("\n=== token counting ===")
    try:
        import tiktoken  # noqa: F401
        print("  tiktoken installed (exact counts)")
    except ImportError:
        print("  tiktoken missing (len//4 fallback)")
    report("estimate_tokens (~5KB)", lambda: estimate_tokens(PROMPT), n=500)


if __name__ == "__main__":
    main()
