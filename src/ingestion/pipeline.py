"""
End-to-end ingestion pipeline.

Orchestrates: load → parse → chunk → embed → store

Usage:
    pipeline = IngestionPipeline()
    stats = pipeline.ingest_directory(Path("./data/raw"))
    print(stats)
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents
from src.ingestion.loaders import load_document
from src.ingestion.parser import parse_documents
from src.utils.files import iter_documents
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class IngestionStats:
    """Summary of an ingestion run."""
    files_attempted: int = 0
    files_succeeded: int = 0
    files_failed: int = 0
    total_pages: int = 0
    total_chunks: int = 0
    failed_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    storage_failed: bool = False
    storage_error: str | None = None

    @property
    def is_fully_successful(self) -> bool:
        """Return True only when processing and persistence both succeeded."""
        return self.files_failed == 0 and not self.storage_failed

    def __str__(self) -> str:
        lines = [
            "─── Ingestion Summary ────────────────",
            f"  Files attempted : {self.files_attempted}",
            f"  Files succeeded : {self.files_succeeded}",
            f"  Files failed    : {self.files_failed}",
            f"  Pages loaded    : {self.total_pages}",
            f"  Chunks produced : {self.total_chunks}",
        ]
        if self.storage_failed:
            lines.append("  Storage status  : failed")
        if self.failed_files:
            lines.append("  Failed files:")
            for f in self.failed_files:
                lines.append(f"    • {f}")
        if self.storage_error:
            lines.append(f"  Storage error   : {self.storage_error}")
        return "\n".join(lines)


class IngestionPipeline:
    """
    Coordinates document ingestion from raw files into the vector store.

    Args:
        vector_store: An initialised VectorStore instance.  If None the
                      pipeline runs in "dry-run" mode — it loads and chunks
                      but does not write to any store.
    """

    def __init__(self, vector_store=None):
        self.vector_store = vector_store

    # ── public API ────────────────────────────────────────────────────────────

    def ingest_file(self, path: Path) -> List[Document]:
        """
        Load, parse, and chunk a single file.

        Returns:
            The list of chunk Documents (not yet stored).

        Raises:
            ValueError: On load or parse failure.
        """
        docs = load_document(path)
        parsed = parse_documents(docs)
        chunks = chunk_documents(parsed)
        return chunks

    def ingest_directory(
        self,
        directory: Path,
        recursive: bool = True,
        file_paths: Optional[List[Path]] = None,
    ) -> IngestionStats:
        """
        Ingest all supported documents from *directory* into the vector store.

        Args:
            directory:   Root directory to scan.
            recursive:   Whether to descend into sub-directories.
            file_paths:  If provided, ingest only these files (ignores directory scan).

        Returns:
            IngestionStats summarising the run.
        """
        stats = IngestionStats()

        paths = file_paths if file_paths is not None else list(iter_documents(directory, recursive))

        if not paths:
            log.warning("No supported documents found in '%s'.", directory)
            return stats

        all_chunks: List[Document] = []

        for path in paths:
            stats.files_attempted += 1
            try:
                chunks = self.ingest_file(path)
                stats.files_succeeded += 1
                stats.total_pages += len(set(c.metadata.get("page", 1) for c in chunks))
                stats.total_chunks += len(chunks)
                all_chunks.extend(chunks)
                log.info("Processed '%s' → %d chunks.", path.name, len(chunks))
            except Exception as exc:
                stats.files_failed += 1
                stats.failed_files.append(path.name)
                stats.errors.append(str(exc))
                log.error("Failed to process '%s': %s", path.name, exc)
                log.debug(traceback.format_exc())

        # Persist to vector store if one is configured
        if self.vector_store is not None and all_chunks:
            log.info("Adding %d chunks to vector store…", len(all_chunks))
            try:
                self.vector_store.add_documents(all_chunks)
            except Exception as exc:
                stats.storage_failed = True
                stats.storage_error = str(exc)
                stats.errors.append(f"Vector store write failed: {exc}")
                log.error("Vector store update failed: %s", exc)
                log.debug(traceback.format_exc())
            else:
                log.info("Vector store updated successfully.")
        elif self.vector_store is None:
            log.info("Dry-run mode: vector store not configured, %d chunks not stored.", len(all_chunks))

        log.info(str(stats))
        return stats
