"""``ragobserve`` command line interface (``ragobserve ui``, like ``mlflow ui``)."""
from __future__ import annotations

import argparse
import os

# Dedicated default port for the RAGObserve dashboard (overridable via
# --port or the RAGOBSERVE_PORT env var). 5601 is RAGObserve's own reserved port,
# kept distinct from MLflow (5000) and other local tooling so it can run
# alongside them without a clash.
DEFAULT_PORT = int(os.getenv("RAGOBSERVE_PORT", "5601"))
DEFAULT_HOST = os.getenv("RAGOBSERVE_HOST", "127.0.0.1")
DEFAULT_STORE = os.getenv("RAGOBSERVE_STORE")  # None -> hidden ./.ragobserve/ragobserve.db


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ragobserve", description="RAGObserve — observability for RAG systems")
    sub = parser.add_subparsers(dest="command")

    ui = sub.add_parser("ui", help="start the RAGObserve tracking server and dashboard")
    ui.add_argument("--host", default=DEFAULT_HOST)
    ui.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"dedicated dashboard port (default: {DEFAULT_PORT}, or $RAGOBSERVE_PORT)")
    ui.add_argument(
        "--backend-store-uri", default=DEFAULT_STORE,
        help="path to the SQLite database file (default: hidden ./.ragobserve/ragobserve.db)",
    )

    prov = sub.add_parser("providers", help="list LLM providers available for generation replay")

    sub.add_parser("version", help="print version")

    args = parser.parse_args(argv)
    if args.command == "version":
        from . import __version__

        print(f"ragobserve {__version__}")
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
        from .storage import resolve

        store = resolve(args.backend_store_uri)
        app = create_app(store)
        print(f"RAGObserve UI: http://{args.host}:{args.port}  (store: {store})")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
