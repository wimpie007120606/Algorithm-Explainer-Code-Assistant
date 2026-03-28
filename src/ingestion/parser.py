"""
Document parser — post-processing layer between raw loaders and the chunker.

Responsibilities:
  1. Basic text normalisation specific to algorithm/CS documents
     (e.g. preserve pseudocode indentation, detect section headings).
  2. Attempt to infer a section heading from the first non-empty line of each
     page/document — stored in metadata["section_heading"].
  3. Flag documents that look like code-heavy pages so the chunker can
     apply a code-aware strategy.

This layer deliberately keeps transformations conservative: we do not want to
destroy structural cues that the LLM will use when composing answers.
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document

from src.utils.logging import get_logger

log = get_logger(__name__)

# Patterns that suggest a section heading (title-case phrase at line start)
_HEADING_RE = re.compile(
    r"^(?:(?:\d+[\.\d]*\s+)?[A-Z][A-Za-z0-9 \-:,/]{3,80})$",
    re.MULTILINE,
)

# A crude heuristic: pages with lots of indented lines are likely pseudocode
_PSEUDOCODE_INDENT_RATIO_THRESHOLD = 0.25


def _infer_section_heading(text: str) -> str:
    """Return the first plausible heading from *text*, or empty string."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and _HEADING_RE.match(stripped):
            return stripped
    return ""


def _is_code_heavy(text: str) -> bool:
    """Return True if the page appears to contain significant code / pseudocode."""
    lines = text.splitlines()
    if not lines:
        return False
    indented = sum(1 for ln in lines if ln.startswith(("    ", "\t")))
    return (indented / len(lines)) >= _PSEUDOCODE_INDENT_RATIO_THRESHOLD


def parse_documents(docs: List[Document]) -> List[Document]:
    """
    Enrich each Document's metadata with inferred structural signals.

    Args:
        docs: Raw documents from the loader layer.

    Returns:
        The same list with updated metadata (in-place mutation for efficiency).
    """
    for doc in docs:
        text = doc.page_content

        heading = _infer_section_heading(text)
        doc.metadata["section_heading"] = heading

        doc.metadata["is_code_heavy"] = _is_code_heavy(text)

        if heading:
            log.debug(
                "Detected heading '%s' on page %s of '%s'.",
                heading,
                doc.metadata.get("page", "?"),
                doc.metadata.get("filename", "?"),
            )

    return docs
