"""
Embedding model factory.

Returns a LangChain-compatible embeddings object based on the configured
provider and model.  The factory is the single place where embedding
dependencies are resolved, keeping the rest of the codebase provider-agnostic.

Supported providers:
  - openai  (default)  — OpenAIEmbeddings
  - gemini             — GoogleGenerativeAIEmbeddings
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

from src.config.settings import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> "Embeddings":
    """
    Return a singleton embedding model for the configured provider.

    The model is cached across calls because constructing it may involve
    network round-trips (API token validation).

    Raises:
        EnvironmentError: If the required API key is missing.
        ValueError:       If the configured provider is not supported.
    """
    settings = get_settings()
    provider = settings.llm_provider
    model_name = settings.embedding_model

    log.info("Initialising embeddings: provider=%s model=%s", provider, model_name)

    if provider == "openai":
        if not settings.openai_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Cannot initialise OpenAI embeddings."
            )
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=settings.openai_api_key,
        )

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Cannot initialise Gemini embeddings."
            )
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is required for Gemini embeddings: "
                "pip install langchain-google-genai"
            ) from exc

        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=settings.gemini_api_key,
        )

    raise ValueError(
        f"Unsupported LLM provider '{provider}'. Supported: openai, gemini"
    )
