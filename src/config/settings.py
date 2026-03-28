"""
Centralised application settings loaded from environment variables / .env file.

All tuneable parameters live here. Import `get_settings()` anywhere in the
codebase — it is cached on first call so there is no repeated I/O.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Load .env from project root (two levels up from this file: src/config/settings.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)


class Settings(BaseSettings):
    """All application settings with defaults and validation."""

    # ── LLM Provider ──────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "gemini"] = Field("openai", alias="LLM_PROVIDER")

    # ── API Keys ──────────────────────────────────────────────────────────────
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")

    # ── Model Selection ───────────────────────────────────────────────────────
    chat_model: str = Field("gpt-4o-mini", alias="CHAT_MODEL")
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")

    # ── Vector Database ───────────────────────────────────────────────────────
    vector_db: Literal["chroma", "pinecone"] = Field("chroma", alias="VECTOR_DB")
    chroma_persist_dir: str = Field("./data/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("algorithm_docs", alias="CHROMA_COLLECTION_NAME")

    # Pinecone (optional)
    pinecone_api_key: str = Field("", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field("algorithm-rag", alias="PINECONE_INDEX_NAME")
    pinecone_environment: str = Field("", alias="PINECONE_ENVIRONMENT")

    # ── Data Paths ────────────────────────────────────────────────────────────
    data_raw_dir: str = Field("./data/raw", alias="DATA_RAW_DIR")
    data_processed_dir: str = Field("./data/processed", alias="DATA_PROCESSED_DIR")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    default_top_k: int = Field(4, alias="DEFAULT_TOP_K")
    similarity_threshold: float = Field(0.3, alias="SIMILARITY_THRESHOLD")

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(150, alias="CHUNK_OVERLAP")

    # ── Generation ────────────────────────────────────────────────────────────
    temperature: float = Field(0.0, alias="TEMPERATURE")
    max_tokens: int = Field(2048, alias="MAX_TOKENS")

    # ── App ───────────────────────────────────────────────────────────────────
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    model_config = {"populate_by_name": True, "env_file": ".env", "extra": "ignore"}

    # ── Derived helpers ───────────────────────────────────────────────────────

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        # Access the chunk_size from the values being validated
        data = info.data
        chunk_size = data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})")
        return v

    def resolved_chroma_dir(self) -> Path:
        """Return an absolute Path for the Chroma persistence directory."""
        p = Path(self.chroma_persist_dir)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p

    def resolved_raw_dir(self) -> Path:
        p = Path(self.data_raw_dir)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p

    def resolved_processed_dir(self) -> Path:
        p = Path(self.data_processed_dir)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p

    def validate_llm_api_key(self) -> None:
        """Raise a clear error if the active provider has no API key configured."""
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file or environment."
            )
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to your .env file or environment."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first load)."""
    return Settings()
