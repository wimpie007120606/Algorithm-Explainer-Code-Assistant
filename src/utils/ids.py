"""
Deterministic ID generation for document chunks.

Using deterministic IDs (rather than random UUIDs) means that re-indexing
the same document produces the same chunk IDs, enabling idempotent upserts.
"""

from __future__ import annotations

import hashlib


def make_chunk_id(source: str, page: int | str, chunk_index: int) -> str:
    """
    Return a stable, unique ID for a chunk.

    Args:
        source:      Original filename or document identifier.
        page:        Page number (or 0 for non-paged documents).
        chunk_index: 0-based index of this chunk within its page/document.

    Returns:
        A 16-character hex string (64-bit collision space — plenty for a KB).
    """
    raw = f"{source}::page={page}::chunk={chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def make_doc_id(source: str) -> str:
    """Return a stable ID for an entire source document."""
    return hashlib.sha256(source.encode()).hexdigest()[:16]
