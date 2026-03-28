"""
Raw document loaders.

Each loader accepts a file path and returns a list of
`langchain_core.documents.Document` objects with populated metadata.

Supported formats:
  - PDF  (.pdf)           — via pypdf
  - Text (.txt)           — plain UTF-8
  - Markdown (.md)        — treated as plain text (structure preserved in content)
  - Images (.png .jpg .jpeg .webp .gif .bmp) — GPT-4o vision description

Design:  The loader layer is intentionally thin.  It only handles I/O and
metadata attachment.  Text cleaning and chunking happen in later stages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.utils.logging import get_logger

log = get_logger(__name__)


# ─── helpers ──────────────────────────────────────────────────────────────────


def _clean_text(text: str) -> str:
    """Remove control characters and normalise whitespace."""
    # Remove null bytes and other binary noise
    text = text.replace("\x00", "")
    # Collapse excessive blank lines (more than two in a row)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─── PDF ──────────────────────────────────────────────────────────────────────


def load_pdf(path: Path) -> List[Document]:
    """
    Load a PDF file, returning one Document per page.

    Each Document carries metadata:
      - source:      absolute file path string
      - filename:    stem + suffix
      - page:        1-based page number
      - total_pages: total pages in the document
      - file_type:   "pdf"

    Raises:
        ImportError: If pypdf is not installed.
        ValueError:  If the PDF cannot be opened or yields no text.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required: pip install pypdf") from exc

    log.info("Loading PDF: %s", path.name)

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{path.name}': {exc}") from exc

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError(f"PDF '{path.name}' has no pages.")

    docs: List[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = _clean_text(raw_text)
        if not text:
            log.warning("Page %d/%d of '%s' yielded no text — skipping.", page_num, total_pages, path.name)
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path.resolve()),
                    "filename": path.name,
                    "page": page_num,
                    "total_pages": total_pages,
                    "file_type": "pdf",
                },
            )
        )

    if not docs:
        raise ValueError(
            f"PDF '{path.name}' produced no extractable text. "
            "It may be scanned/image-only — consider an OCR pre-processor."
        )

    log.info("Loaded %d page(s) from '%s'.", len(docs), path.name)
    return docs


# ─── Plain text ───────────────────────────────────────────────────────────────


def load_text(path: Path) -> List[Document]:
    """
    Load a .txt or .md file as a single Document.

    Metadata:
      - source, filename, page (always 1), total_pages (1), file_type
    """
    log.info("Loading text file: %s", path.name)

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"Cannot read '{path.name}': {exc}") from exc

    text = _clean_text(raw)
    if not text:
        raise ValueError(f"Text file '{path.name}' is empty after cleaning.")

    file_type = "markdown" if path.suffix.lower() == ".md" else "text"

    return [
        Document(
            page_content=text,
            metadata={
                "source": str(path.resolve()),
                "filename": path.name,
                "page": 1,
                "total_pages": 1,
                "file_type": file_type,
            },
        )
    ]


# ─── Image loader (lazy import to avoid requiring OpenAI for text-only use) ───

def _load_image(path: Path) -> List[Document]:
    from src.ingestion.image_loader import load_image
    return load_image(path)


# ─── Dispatcher ───────────────────────────────────────────────────────────────

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

_LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
    **{ext: _load_image for ext in _IMAGE_EXTENSIONS},
}


def load_document(path: Path) -> List[Document]:
    """
    Dispatch to the correct loader based on file extension.

    Args:
        path: Path to the document file.

    Returns:
        List of Document objects (one per page for PDFs, one for images/text).

    Raises:
        ValueError: For unsupported file types or parse failures.
    """
    suffix = path.suffix.lower()
    loader_fn = _LOADERS.get(suffix)
    if loader_fn is None:
        supported = ", ".join(sorted(_LOADERS))
        raise ValueError(
            f"Unsupported file type '{suffix}' for '{path.name}'. "
            f"Supported: {supported}"
        )
    return loader_fn(path)
