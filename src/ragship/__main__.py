"""Command line entry points: `python -m ragship serve` and `python -m ragship eval`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ragship.config import load_settings
from ragship.eval.harness import format_report, run_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ragship",
        description="Retrieval-augmented generation over documents you ingest.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve_parser = sub.add_parser("serve", help="Run the HTTP API")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)

    eval_parser = sub.add_parser("eval", help="Measure Recall@k on a labeled set")
    eval_parser.add_argument("--k", type=int, default=None)
    eval_parser.add_argument("--dataset", type=Path, default=None)
    eval_parser.add_argument("--corpus", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "serve":
        from ragship.api.app import serve

        serve(host=args.host, port=args.port)
        return 0

    if args.k is not None and args.k < 1:
        parser.error("--k must be >= 1")

    settings = load_settings()
    try:
        report = run_eval(
            k=args.k,
            dataset=args.dataset,
            corpus=args.corpus,
            settings=settings,
        )
    except (OSError, ValueError) as exc:
        print(f"ragship: {exc}", file=sys.stderr)
        return 2
    print(format_report(report))
    if report.recall + 1e-9 < settings.min_recall:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
