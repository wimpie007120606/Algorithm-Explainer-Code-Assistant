"""
Run the retrieval evaluation suite and print a report.

Usage:
    python scripts/run_eval.py [--top-k K] [--dry-run] [--output FILE]

Options:
    --top-k K       Chunks to retrieve per question (default: 4)
    --dry-run       Skip LLM calls — evaluate retrieval only
    --output FILE   Save report as JSON to this file path
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.eval_runner import EvalRunner
from src.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate retrieval only — do not call the LLM.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save report JSON.",
    )
    args = parser.parse_args()

    log.info("Running evaluation: top_k=%d dry_run=%s", args.top_k, args.dry_run)

    runner = EvalRunner(top_k=args.top_k, dry_run=args.dry_run)
    report = runner.run()
    report.print_report()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
