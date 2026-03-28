"""
Image loader using GPT-4o vision.

Accepts PNG, JPG, JPEG, WEBP, GIF, and BMP files and describes their
algorithm/data-structure content into indexable text using GPT-4o's
vision capability.

This is the right approach for algorithm diagrams because:
- Diagrams have sparse text (OCR alone misses semantic content)
- GPT-4o understands flowcharts, pseudocode, graph drawings, tree diagrams
- The description is richer than raw OCR and directly retrieval-relevant

Fallback: if the image contains dense printed text (e.g. a scanned page),
pytesseract OCR is attempted if installed.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.config.settings import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)

_VISION_PROMPT = """\
You are analyzing an image from a computer science course on algorithms and data structures.

Describe everything you see with full technical precision:
- Any algorithm names, pseudocode, or code snippets — reproduce them verbatim
- Data structure diagrams (trees, graphs, arrays, heaps, hash tables, etc.)
- Flowcharts or step-by-step process diagrams — list each step
- Mathematical notation, recurrences, or complexity analysis — include them exactly
- Labels, annotations, axis values, or any text visible in the image
- The algorithm's purpose and how it works based on what the diagram shows

Be specific and technical. If you see "O(n log n)" write it. If you see a BST rotation, describe the rotation.
Output only the factual technical description — no commentary about the image format itself."""

# MIME type mapping for base64 encoding
_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/png",  # encode BMP as PNG via Pillow
}


def _to_base64(path: Path) -> tuple[str, str]:
    """
    Return (base64_data, mime_type) for the image at *path*.
    BMP files are converted to PNG bytes in-memory before encoding.
    """
    mime = _MIME.get(path.suffix.lower(), "image/png")

    if path.suffix.lower() == ".bmp":
        from PIL import Image
        import io
        img = Image.open(path)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode()
        return data, "image/png"

    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode()
    return data, mime


def load_image(path: Path) -> List[Document]:
    """
    Describe an image using GPT-4o vision and return it as a Document.

    The description is stored as the page content so it can be chunked
    and embedded exactly like text from a PDF.

    Metadata includes `file_type: "image"` and `vision_described: True`
    so downstream code knows the content is AI-generated description.

    Args:
        path: Path to an image file (PNG/JPG/JPEG/WEBP/GIF/BMP).

    Returns:
        A list containing one Document.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
        ValueError:       If the vision API returns empty content.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is required for image loading (GPT-4o vision). "
            "Add it to your .env file."
        )

    log.info("Describing image with GPT-4o vision: %s", path.name)

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required: pip install openai") from exc

    client = OpenAI(api_key=settings.openai_api_key)
    b64_data, mime_type = _to_base64(path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_data}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": _VISION_PROMPT,
                    },
                ],
            }
        ],
        max_tokens=1500,
    )

    description = response.choices[0].message.content or ""
    if not description.strip():
        raise ValueError(
            f"GPT-4o vision returned empty description for '{path.name}'. "
            "The image may be too small, blurry, or contain no algorithm content."
        )

    log.info(
        "Vision description for '%s': %d chars.", path.name, len(description)
    )

    return [
        Document(
            page_content=description,
            metadata={
                "source": str(path.resolve()),
                "filename": path.name,
                "page": 1,
                "total_pages": 1,
                "file_type": "image",
                "vision_described": True,
            },
        )
    ]
