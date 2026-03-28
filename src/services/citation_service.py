"""
CitationService — builds source citation summaries from retrieved chunks.

Responsible for:
  - Deduplicating sources (same filename, same page → one citation)
  - Formatting citations for display in the UI
  - Producing a markdown-formatted reference list
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.vectordb.retriever import RetrievedChunk
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Citation:
    """A single source citation."""

    filename: str
    page: int | str
    score: float
    chunk_preview: str  # First 200 chars of the chunk content
    chunk_id: str

    def as_markdown(self) -> str:
        """Return a compact markdown citation string."""
        score_pct = f"{self.score * 100:.0f}%"
        return (
            f"**{self.filename}** — Page {self.page} "
            f"*(relevance: {score_pct})*\n"
            f"> {self.chunk_preview}…"
        )


class CitationService:
    """Builds and formats citations from a list of RetrievedChunks."""

    PREVIEW_LENGTH = 200

    @classmethod
    def build_citations(cls, chunks: List[RetrievedChunk]) -> List[Citation]:
        """
        Convert retrieved chunks into deduplicated Citation objects.

        Two chunks from the same filename+page are merged into a single
        citation (the one with the higher score is kept).

        Args:
            chunks: Retrieved chunks from VectorStoreRetriever.

        Returns:
            List of Citation objects sorted by relevance score (best first).
        """
        seen: dict[tuple, Citation] = {}

        for chunk in chunks:
            meta = chunk.metadata
            filename = meta.get("filename", "Unknown")
            page = meta.get("page", "N/A")
            key = (filename, str(page))

            preview = chunk.content.strip()[: cls.PREVIEW_LENGTH].replace("\n", " ")

            citation = Citation(
                filename=filename,
                page=page,
                score=chunk.score,
                chunk_preview=preview,
                chunk_id=meta.get("chunk_id", ""),
            )

            if key not in seen or chunk.score > seen[key].score:
                seen[key] = citation

        citations = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        log.debug("Built %d citation(s) from %d chunks.", len(citations), len(chunks))
        return citations

    @classmethod
    def format_markdown(cls, citations: List[Citation]) -> str:
        """Return a markdown-formatted reference section."""
        if not citations:
            return "*No sources retrieved.*"

        lines = ["### Sources\n"]
        for i, cite in enumerate(citations, start=1):
            lines.append(f"{i}. {cite.as_markdown()}\n")

        return "\n".join(lines)

    @classmethod
    def format_inline_references(cls, citations: List[Citation]) -> str:
        """Return a compact inline reference list (e.g. for answer suffix)."""
        if not citations:
            return ""
        refs = [f"{c.filename} p.{c.page}" for c in citations]
        return "Sources: " + " | ".join(refs)
