"""
File system helpers used across the ingestion and storage pipelines.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Iterator

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def iter_documents(directory: Path, recursive: bool = True) -> Iterator[Path]:
    """
    Yield all supported document files under *directory*.

    Args:
        directory: Root directory to scan.
        recursive: If True (default), scan sub-directories.

    Yields:
        Absolute Path objects for each supported file.

    Raises:
        FileNotFoundError: If *directory* does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    pattern = "**/*" if recursive else "*"
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def file_hash(path: Path, algorithm: str = "md5") -> str:
    """Return a hex digest of *path* contents — used for deduplication."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it does not exist, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_copy(src: Path, dst_dir: Path) -> Path:
    """
    Copy *src* into *dst_dir*, creating the directory if needed.

    If a file with the same name already exists at the destination, it is
    overwritten only when the content hash differs.

    Returns:
        The destination path.
    """
    ensure_dir(dst_dir)
    dst = dst_dir / src.name

    if dst.exists() and file_hash(src) == file_hash(dst):
        return dst  # identical file already present

    shutil.copy2(src, dst)
    return dst
