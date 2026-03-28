"""
Rebuild the vector index from scratch.

This script:
  1. Deletes all documents from the vector store (reset).
  2. Re-ingests all documents from the raw data directory.

Use this when you want to re-chunk with new settings or after updating documents.

Usage:
    python scripts/rebuild_index.py [--dir PATH] [--yes]

Options:
    --dir PATH   Source directory (default: DATA_RAW_DIR from settings)
    --yes        Skip confirmation prompt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_settings
from src.ingestion.pipeline import IngestionPipeline
from src.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wipe and rebuild the RAG vector index."
    )
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()

    settings = get_settings()
    directory = args.dir or settings.resolved_raw_dir()

    if not args.yes:
        answer = input(
            f"This will DELETE all indexed documents and re-ingest from '{directory}'.\n"
            "Continue? [y/N] "
        ).strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    try:
        from src.vectordb.retriever import get_vector_store

        vector_store = get_vector_store(reset=True)
    except Exception as exc:
        print(f"[ERROR] Could not connect to vector store: {exc}")
        sys.exit(1)

    pipeline = IngestionPipeline(vector_store=vector_store)
    stats = pipeline.ingest_directory(directory)
    print(str(stats))

    if stats.files_failed > 0:
        sys.exit(1)

    print("\n✓ Index rebuilt successfully.")


if __name__ == "__main__":
    main()
