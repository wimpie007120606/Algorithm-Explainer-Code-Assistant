"""
Logging configuration for the entire application.

Call `get_logger(__name__)` at the top of every module.  The root logger is
configured once on first import; subsequent calls are no-ops.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_ROOT_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger. Safe to call multiple times."""
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _ROOT_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger, configuring the root logger on first call."""
    # Lazily read log level from settings to avoid circular imports
    try:
        from src.config.settings import get_settings

        level = get_settings().log_level
    except Exception:
        level = "INFO"

    configure_logging(level)
    return logging.getLogger(name or __name__)
