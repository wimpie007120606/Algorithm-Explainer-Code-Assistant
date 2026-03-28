"""
Ingest documents from the configured data/raw directory into the vector store.

Usage:
    python scripts/ingest_docs.py [--dir PATH] [--reset] [--dry-run]

Options:
    --dir PATH    Directory to ingest (default: DATA_RAW_DIR from settings)
    --reset       Wipe the vector store before ingesting (full rebuild)
    --dry-run     Load and chunk documents but do not write to the vector store
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_settings
from src.ingestion.pipeline import IngestionPipeline
from src.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory containing documents to ingest (default: DATA_RAW_DIR).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the vector store before ingesting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk documents but do not write to the vector store.",
    )
    args = parser.parse_args()

    settings = get_settings()
    directory = args.dir or settings.resolved_raw_dir()

    if not directory.exists():
        print(f"[ERROR] Directory not found: {directory}")
        print(f"        Create it and place your PDFs/TXT/MD files there.")
        sys.exit(1)

    log.info("Ingestion starting. Source: %s | reset=%s | dry_run=%s", directory, args.reset, args.dry_run)

    # Vector store (skip in dry-run mode)
    vector_store = None
    if not args.dry_run:
        try:
            from src.vectordb.retriever import get_vector_store

            vector_store = get_vector_store(reset=args.reset)
        except Exception as exc:
            print(f"[ERROR] Could not connect to vector store: {exc}")
            print("        Check your .env configuration.")
            sys.exit(1)

    pipeline = IngestionPipeline(vector_store=vector_store)
    stats = pipeline.ingest_directory(directory)

    print(str(stats))

    if stats.files_failed > 0:
        print("\n[WARNING] Some files failed — check the log above for details.")
        sys.exit(1)

    print("\n✓ Ingestion complete.")


if __name__ == "__main__":
    main()
