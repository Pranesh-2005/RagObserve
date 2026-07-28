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


def main(argv=None) -> int:
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
    exp = sub.add_parser("export", help="export traces to NDJSON")
    exp.add_argument("--project", required=True, metavar="NAME")
    exp.add_argument("--output", "-o", default="-", metavar="FILE",
                     help="output file path (default: stdout)")
    exp.add_argument("--limit", type=int, default=10_000)
    _store_arg(exp)

    # ── eval ────────────────────────────────────────────────────────────────
    ev = sub.add_parser("eval", help="run LLM-as-judge faithfulness scoring on a project")
    ev.add_argument("--project", required=True, metavar="NAME")
    ev.add_argument("--model", default="llama3-8b-8192",
                    help="Groq model (default: llama3-8b-8192)")
    ev.add_argument("--limit", type=int, default=50,
                    help="max traces to score (default: 50)")
    ev.add_argument("--api-key", default=os.getenv("GROQ_API_KEY", ""),
                    help="Groq API key (default: $GROQ_API_KEY)")
    _store_arg(ev)

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
        from .storage import resolve

        if args.key:
            os.environ["RAGOBSERVE_API_KEY"] = args.key

        key = get_api_key()
        store_path = resolve(args.backend_store_uri)
        app = create_app(store_path)

        print(f"RAGObserve UI  → http://{args.host}:{args.port}/?key={key}")
        print(f"API key        → {key}")
        print(f"Store          → {store_path}")
        print("Set RAGOBSERVE_API_KEY env var to use a fixed key.")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        return 0

    if args.command == "export":
        import json
        from .storage import resolve

        uri = args.backend_store_uri or ""
        if uri.startswith("postgresql://") or uri.startswith("postgres://"):
            from .stores import PostgresStore
            store = PostgresStore(uri)
        else:
            from .server.db import Store
            store = Store(resolve(uri or None))
        traces = store.list_traces(args.project, limit=args.limit)

        out = open(args.output, "w", encoding="utf-8") if args.output != "-" else sys.stdout
        try:
            for t in traces:
                detail = store.get_trace(t["trace_id"])
                if detail:
                    out.write(json.dumps(detail, default=str) + "\n")
        finally:
            if args.output != "-":
                out.close()

        count = len(traces)
        if args.output != "-":
            print(f"Exported {count} traces → {args.output}")
        return 0

    if args.command == "eval":
        from .eval import evaluate_trace
        from .storage import resolve

        if args.api_key:
            os.environ["GROQ_API_KEY"] = args.api_key

        uri = args.backend_store_uri or ""
        if uri.startswith("postgresql://") or uri.startswith("postgres://"):
            from .stores import PostgresStore
            store = PostgresStore(uri)
        else:
            from .server.db import Store
            store = Store(resolve(uri or None))
        traces = store.list_traces(args.project, limit=args.limit)

        if not traces:
            print(f"No traces found for project '{args.project}'")
            return 1

        ok = 0
        for t in traces:
            detail = store.get_trace(t["trace_id"])
            if detail is None:
                continue
            try:
                scores = evaluate_trace(detail, model=args.model)
                for metric, result in scores.items():
                    store.set_eval_score(
                        t["trace_id"], args.project, metric,
                        result.get("score"), result.get("reason", ""), args.model,
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

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
