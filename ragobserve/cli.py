"""``ragobserve`` command line interface."""
from __future__ import annotations

import argparse
import os
import sys

DEFAULT_PORT = int(os.getenv("RAGOBSERVE_PORT", "5601"))
DEFAULT_HOST = os.getenv("RAGOBSERVE_HOST", "127.0.0.1")
DEFAULT_STORE = os.getenv("RAGOBSERVE_STORE")


def _store_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--backend-store-uri", default=DEFAULT_STORE, metavar="PATH",
        help="SQLite path or postgresql:// DSN (default: .ragobserve/ragobserve.db)",
    )


def _open_store(uri):
    """Postgres for a postgres:// DSN, else SQLite at the path (or the default)."""
    uri = uri or ""
    if uri.startswith(("postgresql://", "postgres://")):
        from .stores import PostgresStore
        return PostgresStore(uri)
    from .server.db import Store
    from .storage import resolve
    return Store(resolve(uri or None))


def main(argv=None) -> int:
    # Windows pipes/redirects default to cp1252, which can't encode the → / ✓ we
    # print, so `ragobserve ui > log.txt` crashed before the server started.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        prog="ragobserve",
        description="RAGObserve — local-first observability for RAG systems",
    )
    sub = parser.add_subparsers(dest="command")

    # ── ui ──────────────────────────────────────────────────────────────────
    ui = sub.add_parser("ui", help="start the dashboard server")
    ui.add_argument("--host", default=DEFAULT_HOST)
    ui.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"port (default: {DEFAULT_PORT}, or $RAGOBSERVE_PORT)")
    ui.add_argument("--key", default=os.getenv("RAGOBSERVE_API_KEY", ""), metavar="KEY",
                    help="API key (auto-generated and printed if not set)")
    _store_arg(ui)

    # ── export ──────────────────────────────────────────────────────────────
    exp = sub.add_parser("export", help="export traces to NDJSON (raw traces or Ragas samples)")
    exp.add_argument("--project", required=True, metavar="NAME")
    exp.add_argument("--format", choices=["raw", "ragas"], default="raw",
                     help="raw = full trace JSON; ragas = one Ragas sample per line "
                          "(user_input, retrieved_contexts, response, ...)")
    exp.add_argument("--output", "-o", default="-", metavar="FILE",
                     help="output file path (default: stdout)")
    exp.add_argument("--limit", type=int, default=10_000)
    _store_arg(exp)

    # ── eval ────────────────────────────────────────────────────────────────
    ev = sub.add_parser("eval", help="score faithfulness + relevance on a project")
    ev.add_argument("--project", required=True, metavar="NAME")
    ev.add_argument("--backend", choices=["judge", "ragas"], default="judge",
                    help="judge = built-in LLM judge (default); ragas = Ragas metrics")
    ev.add_argument("--provider", default="groq",
                    help="judge provider, any from `ragobserve providers` (default: groq)")
    ev.add_argument("--model", default=None,
                    help="judge model (default: the provider's default model)")
    ev.add_argument("--limit", type=int, default=50,
                    help="max traces to score (default: 50)")
    ev.add_argument("--api-key", default="",
                    help="judge provider API key (default: the provider's env var)")
    _store_arg(ev)

    # ── test (golden-set regression) ────────────────────────────────────────
    ts = sub.add_parser("test", help="run a golden question set through your pipeline; "
                                     "exit 1 if quality drops below thresholds")
    ts.add_argument("--golden", required=True, metavar="FILE",
                    help="JSONL: {question, relevant_chunk_ids?, reference?} per line")
    ts.add_argument("--pipeline", required=True, metavar="MODULE:FN",
                    help="your instrumented fn(question) -> answer, e.g. app.rag:answer")
    ts.add_argument("--project", default=None,
                    help="project for this run (default: golden-<timestamp>)")
    ts.add_argument("--k", type=int, default=10)
    for name in ("recall", "precision", "mrr", "ndcg", "faithfulness", "answer_relevance"):
        ts.add_argument(f"--min-{name.replace('_', '-')}", type=float, default=None, metavar="X")
    ts.add_argument("--judge-provider", default=None,
                    help="score faithfulness/relevance with the LLM judge on this provider "
                         "(implied as groq by --min-faithfulness / --min-answer-relevance)")
    ts.add_argument("--judge-model", default=None)
    ts.add_argument("--json", action="store_true", help="print the report as JSON")
    _store_arg(ts)

    # ── import-ragas ────────────────────────────────────────────────────────
    ir = sub.add_parser("import-ragas", help="attach scores from a Ragas results CSV/JSONL to traces")
    ir.add_argument("file", metavar="FILE", help="result.to_pandas().to_csv(...) output, or JSONL")
    ir.add_argument("--project", required=True, metavar="NAME")
    _store_arg(ir)

    # ── providers ───────────────────────────────────────────────────────────
    sub.add_parser("providers", help="list LLM providers available for generation replay")

    # ── prices ──────────────────────────────────────────────────────────────
    pr = sub.add_parser("prices", help="show or refresh the model price book")
    pr.add_argument("--refresh", action="store_true",
                    help="download the latest all-provider price feed")
    pr.add_argument("--model", metavar="NAME", help="look up one model's rate")

    # ── version ─────────────────────────────────────────────────────────────
    sub.add_parser("version", help="print version and exit")

    args = parser.parse_args(argv)

    # ── dispatch ────────────────────────────────────────────────────────────

    if args.command == "version":
        from . import __version__
        print(f"ragobserve {__version__}")
        return 0

    if args.command == "prices":
        import datetime
        from .server import pricing

        if args.refresh:
            try:
                n = pricing.refresh()
            except Exception as e:
                print(f"price refresh failed: {e}", file=sys.stderr)
                return 1
            print(f"refreshed {n} models -> {pricing.cache_path()}")

        if args.model:
            price = pricing._lookup(args.model)
            if price is None:
                print(f"{args.model}: unknown (no cost will be estimated)")
                return 1
            print(f"{args.model}: ${price[0]:g} in / ${price[1]:g} out  per 1M tokens")
            return 0

        info = pricing.feed_info()
        if info:
            when = datetime.datetime.fromtimestamp(info["updated_at"]).strftime("%Y-%m-%d %H:%M")
            print(f"feed     {info['count']} models, updated {when}")
            print(f"source   {info['source']}")
        else:
            print("feed     not downloaded — run `ragobserve prices --refresh`")
        print(f"builtin  {len(pricing.PRICE_BOOK)} models (offline fallback)")
        return 0

    if args.command == "providers":
        from .server import llm
        for p in llm.available_providers():
            mark = "✓" if p["ready"] else "·"
            print(f"  {mark} {p['id']:<12} {p['label']:<22} {'ready' if p['ready'] else p['hint']}")
        return 0

    if args.command == "ui":
        import uvicorn
        from .server.app import create_app
        from .server.auth import get_api_key

        if args.key:
            os.environ["RAGOBSERVE_API_KEY"] = args.key

        key = get_api_key()
        store = _open_store(args.backend_store_uri)
        app = create_app(store=store)

        print(f"RAGObserve UI  → http://{args.host}:{args.port}/?key={key}")
        print(f"API key        → {key}")
        print(f"Store          → {getattr(store, 'path', None) or 'postgres'}")
        print("Set RAGOBSERVE_API_KEY env var to use a fixed key.")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        return 0

    if args.command == "export":
        import json

        store = _open_store(args.backend_store_uri)
        traces = store.list_traces(args.project, limit=args.limit)

        if args.format == "ragas":
            from .ragas import trace_to_sample
            fmt = trace_to_sample
        else:
            fmt = lambda d: d  # noqa: E731

        out = open(args.output, "w", encoding="utf-8") if args.output != "-" else sys.stdout
        try:
            for t in traces:
                detail = store.get_trace(t["trace_id"])
                if detail:
                    out.write(json.dumps(fmt(detail), default=str) + "\n")
        finally:
            if args.output != "-":
                out.close()

        count = len(traces)
        if args.output != "-":
            print(f"Exported {count} traces → {args.output}")
        return 0

    if args.command == "eval":
        from .eval import evaluate_trace
        from .server import llm

        store = _open_store(args.backend_store_uri)
        traces = store.list_traces(args.project, limit=args.limit)

        if not traces:
            print(f"No traces found for project '{args.project}'")
            return 1

        if args.backend == "ragas":
            from .ragas import evaluate_trace as ragas_eval
            model = "ragas"
            score = ragas_eval
        else:
            try:
                model = args.model or llm.default_model(args.provider)
            except llm.ProviderError as e:
                print(str(e), file=sys.stderr)
                return 1

            def score(detail):
                return evaluate_trace(detail, model=model, api_key=args.api_key,
                                      provider=args.provider)

        ok = 0
        for t in traces:
            detail = store.get_trace(t["trace_id"])
            if detail is None:
                continue
            try:
                scores = score(detail)
                for metric, result in scores.items():
                    store.set_eval_score(
                        t["trace_id"], args.project, metric,
                        result.get("score"), result.get("reason", ""), model,
                    )
                faith = scores["faithfulness"]["score"]
                rel   = scores["answer_relevance"]["score"]
                f_str = f"{faith:.2f}" if faith is not None else "n/a"
                r_str = f"{rel:.2f}"   if rel   is not None else "n/a"
                print(f"  {t['trace_id'][:12]}  faithfulness={f_str}  relevance={r_str}")
                ok += 1
            except Exception as e:
                print(f"  {t['trace_id'][:12]}  ERROR: {e}", file=sys.stderr)

        print(f"\nScored {ok}/{len(traces)} traces for project '{args.project}'")
        return 0

    if args.command == "test":
        import json
        from . import golden

        try:
            rows = golden.load_golden(args.golden)
            pipeline = golden.load_pipeline(args.pipeline)
        except (OSError, ValueError, ImportError, AttributeError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        minimums = {n: getattr(args, f"min_{n}") for n in golden.THRESHOLDS}
        provider = args.judge_provider or (
            "groq" if minimums["faithfulness"] is not None or minimums["answer_relevance"] is not None
            else None)
        judge = {"provider": provider, "model": args.judge_model} if provider else None
        project = args.project or golden.default_project()
        store = _open_store(args.backend_store_uri)
        report = golden.run(pipeline, rows, store, project, k=args.k, judge=judge)
        fails = golden.check(report, minimums)
        if args.json:
            print(json.dumps({**report, "failures": fails}, indent=2, default=str))
        else:
            def f(v):
                return "n/a" if v is None else f"{v:.3f}"
            print(f"project  {project}  ({report['questions']} questions, k={args.k})")
            for n in golden.THRESHOLDS:
                floor = minimums[n]
                print(f"  {n:<17}{f(report[n]):>8}" + (f"   min {floor}" if floor is not None else ""))
            for e in report["errors"]:
                print(f"  ERROR  {e['question'][:60]}: {e['error']}", file=sys.stderr)
            print("FAIL\n  " + "\n  ".join(fails) if fails else "PASS")
        return 1 if fails else 0

    if args.command == "import-ragas":
        from .ragas import import_results, read_results

        store = _open_store(args.backend_store_uri)
        res = import_results(store, args.project, read_results(args.file))
        print(f"matched {res['matched']} rows, stored {res['scores']} scores")
        for u in res["unmatched"][:10]:
            print(f"  unmatched: {str(u)[:80]}", file=sys.stderr)
        return 0 if res["matched"] else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
