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
import unicodedata
from pathlib import Path

from langchain_core.documents import Document

from src.config.settings import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)


# ─── helpers ──────────────────────────────────────────────────────────────────

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_BOILERPLATE_LINE_RE = re.compile(
    r"("
    r"copyright\s+\d{4}.*all\s+rights\s+reserved"
    r"|all\s+rights\s+reserved.*may\s+not\s+be\s+copied"
    r"|may\s+not\s+be\s+copied,\s*scanned,\s*or\s+duplicated"
    r"|editorial\s+review\s+has\s+deemed"
    r"|does\s+not\s+materially\s+affect\s+the\s+overall\s+learning\s+experience"
    r"|cengage\s+learning\s+reserves"
    r"|registered\s+trademarks?\s+of"
    r"|cyanmagentayellowblack"
    r"|sheet\s+number\s+\d+\s+page\s+number"
    r")",
    re.IGNORECASE,
)


def _is_boilerplate_line(line: str) -> bool:
    """Return True for publisher/footer/legal lines that hurt retrieval."""
    compact = re.sub(r"\s+", " ", line).strip()
    return bool(_BOILERPLATE_LINE_RE.search(compact))


def _clean_text(text: str) -> str:
    """Remove extraction noise and normalise whitespace while preserving study content."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHAR_RE.sub("", text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line and _is_boilerplate_line(line):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    # Collapse excessive blank lines (more than two in a row)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _has_substantive_text(text: str, min_chars: int) -> bool:
    """Return True when extracted text looks useful enough to index."""
    compact = re.sub(r"\s+", " ", text).strip()
    alpha_numeric = re.findall(r"[A-Za-z0-9]", compact)
    if len(alpha_numeric) < min_chars:
        return False
    if re.fullmatch(r"(?i)page\s+\d+(?:\s+of\s+\d+)?", compact):
        return False
    if _is_boilerplate_line(compact):
        return False
    return True


def _render_pdf_page_to_png(path: Path, page_index: int, dpi: int) -> bytes:
    """Render one PDF page to PNG bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF vision fallback: pip install PyMuPDF"
        ) from exc

    zoom = dpi / 72
    with fitz.open(str(path)) as doc:
        page = doc.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pixmap.tobytes("png")


def _describe_pdf_page_with_vision(
    path: Path,
    *,
    page_index: int,
    page_num: int,
    total_pages: int,
) -> str:
    """Use GPT-4o vision to transcribe a PDF page that has little text."""
    settings = get_settings()
    png_bytes = _render_pdf_page_to_png(path, page_index, settings.pdf_vision_dpi)

    from src.ingestion.image_loader import describe_image_bytes

    prompt = f"""\
You are extracting study content from page {page_num} of {total_pages} in a PDF named {path.name}.

Transcribe the page as faithfully as possible for retrieval:
- Preserve question numbers, sub-questions, answer choices, mark allocations, headings, and instructions.
- Preserve formulas, symbols, integrals, derivatives, limits, matrices, units, diagrams, tables, and worked steps.
- For visual diagrams, describe labels, axes, shapes, arrows, relationships, and any text in the diagram.
- If handwriting or scanned text is unclear, write [unclear] for that span rather than guessing.

Output only source content and concise visual descriptions."""

    return describe_image_bytes(
        png_bytes,
        "image/png",
        source_label=f"{path.name} page {page_num}",
        prompt=prompt,
        max_tokens=2600,
    )


# ─── PDF ──────────────────────────────────────────────────────────────────────


def load_pdf(path: Path) -> list[Document]:
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

    settings = get_settings()
    log.info("Loading PDF: %s", path.name)

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{path.name}': {exc}") from exc

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError(f"PDF '{path.name}' has no pages.")

    docs: list[Document] = []
    skipped_pages: list[int] = []
    vision_pages: list[int] = []

    for page_index, page in enumerate(reader.pages):
        page_num = page_index + 1
        raw_text = page.extract_text() or ""
        text = _clean_text(raw_text)

        extraction_method = "pypdf"
        vision_described = False

        if not _has_substantive_text(text, settings.pdf_min_text_chars):
            if settings.pdf_vision_fallback and settings.openai_api_key:
                try:
                    text = _clean_text(
                        _describe_pdf_page_with_vision(
                            path,
                            page_index=page_index,
                            page_num=page_num,
                            total_pages=total_pages,
                        )
                    )
                    extraction_method = "gpt4o_vision_pdf_page"
                    vision_described = True
                    vision_pages.append(page_num)
                except Exception as exc:
                    log.warning(
                        "Page %d/%d of '%s' could not use PDF vision fallback: %s",
                        page_num,
                        total_pages,
                        path.name,
                        exc,
                    )
                    text = ""

            if not _has_substantive_text(text, settings.pdf_min_text_chars):
                skipped_pages.append(page_num)
                log.warning(
                    "Page %d/%d of '%s' yielded no substantive text — skipping.",
                    page_num,
                    total_pages,
                    path.name,
                )
                continue

        if not text:
            skipped_pages.append(page_num)
            log.warning(
                "Page %d/%d of '%s' yielded no text — skipping.",
                page_num,
                total_pages,
                path.name,
            )
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
                    "extraction_method": extraction_method,
                    "vision_described": vision_described,
                },
            )
        )

    if not docs:
        fallback_hint = (
            "Install PyMuPDF and keep PDF_VISION_FALLBACK=true to let GPT-4o vision "
            "transcribe scanned/math-heavy pages."
            if settings.openai_api_key
            else "Add OPENAI_API_KEY to enable GPT-4o vision fallback for scanned/math-heavy pages."
        )
        raise ValueError(
            f"PDF '{path.name}' produced no substantive extractable text. "
            f"It may be scanned/image-only or math rendered as images. {fallback_hint}"
        )

    if skipped_pages:
        log.warning(
            "Skipped %d low-text page(s) in '%s': %s",
            len(skipped_pages),
            path.name,
            ", ".join(str(p) for p in skipped_pages),
        )
    if vision_pages:
        log.info(
            "Used PDF vision fallback for %d page(s) in '%s': %s",
            len(vision_pages),
            path.name,
            ", ".join(str(p) for p in vision_pages),
        )

    log.info("Loaded %d page(s) from '%s'.", len(docs), path.name)
    return docs


# ─── Plain text ───────────────────────────────────────────────────────────────


def load_text(path: Path) -> list[Document]:
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

def _load_image(path: Path) -> list[Document]:
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


def load_document(path: Path) -> list[Document]:
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
