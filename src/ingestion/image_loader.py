"""
Image loader using GPT-4o vision.

Accepts PNG, JPG, JPEG, WEBP, GIF, and BMP files and describes their study
content into indexable text using GPT-4o's vision capability.

This helps with scanned notes, diagrams, math-heavy pages, tables, charts,
annotated slides, and photographed question papers where plain text extraction
is weak or unavailable.
"""

from __future__ import annotations

import base64
from pathlib import Path

from langchain_core.documents import Document

from src.config.settings import get_settings, missing_secret_message
from src.utils.logging import get_logger

log = get_logger(__name__)

_VISION_PROMPT = """\
You are extracting study material from an uploaded page or image.

Transcribe and describe everything that would help a student study from this source:
- Reproduce visible text, headings, question numbers, answers, equations, and labels as exactly as possible.
- Preserve mathematical notation, units, symbols, tables, charts, diagrams, code, and step-by-step workings.
- For diagrams or visual layouts, explain the relationships, labels, axes, arrows, and important visual cues.
- If this is an exam, worksheet, memo, or solution page, keep question numbers and mark allocations when visible.
- If handwriting or scanned text is unclear, mark the uncertain span as [unclear] rather than guessing.

Output only factual source content and concise descriptions. Do not add advice, solve new problems, or mention image quality."""

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
        import io

        from PIL import Image
        img = Image.open(path)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode()
        return data, "image/png"

    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode()
    return data, mime


def describe_image_bytes(
    image_bytes: bytes,
    mime_type: str,
    *,
    source_label: str,
    prompt: str = _VISION_PROMPT,
    max_tokens: int = 2200,
) -> str:
    """Return a factual text description/transcription for image bytes."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise OSError(
            f"{missing_secret_message('OPENAI_API_KEY')} "
            "Vision ingestion uses GPT-4o even if the main chat provider is Gemini."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required: pip install openai") from exc

    log.info("Describing visual content with GPT-4o vision: %s", source_label)

    client = OpenAI(api_key=settings.openai_api_key)
    b64_data = base64.b64encode(image_bytes).decode()

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
                        "text": prompt,
                    },
                ],
            }
        ],
        max_tokens=max_tokens,
    )

    description = response.choices[0].message.content or ""
    if not description.strip():
        raise ValueError(
            f"GPT-4o vision returned empty content for '{source_label}'. "
            "The image may be too small, blurry, or contain no readable study material."
        )

    log.info("Vision description for '%s': %d chars.", source_label, len(description))
    return description.strip()


def load_image(path: Path) -> list[Document]:
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
        raise OSError(
            f"{missing_secret_message('OPENAI_API_KEY')} "
            "Image ingestion uses GPT-4o vision even if the main chat provider is Gemini."
        )

    b64_data, mime_type = _to_base64(path)
    description = describe_image_bytes(
        base64.b64decode(b64_data),
        mime_type,
        source_label=path.name,
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
