"""
LLM factory.

Returns a LangChain chat model object for the configured provider.
Centralising model construction here means the rest of the codebase never
imports from provider-specific packages directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

from src.config.settings import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> "BaseChatModel":
    """
    Return a singleton chat model for the configured provider.

    Raises:
        EnvironmentError: API key missing.
        ValueError:       Unsupported provider.
        ImportError:      Provider-specific package not installed.
    """
    settings = get_settings()
    provider = settings.llm_provider

    log.info(
        "Initialising LLM: provider=%s model=%s temperature=%s",
        provider,
        settings.chat_model,
        settings.temperature,
    )

    if provider == "openai":
        if not settings.openai_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.chat_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            openai_api_key=settings.openai_api_key,
        )

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is required: pip install langchain-google-genai"
            ) from exc

        return ChatGoogleGenerativeAI(
            model=settings.chat_model,
            temperature=settings.temperature,
            google_api_key=settings.gemini_api_key,
        )

    raise ValueError(
        f"Unsupported LLM provider '{provider}'. Supported: openai, gemini"
    )
