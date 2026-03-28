"""
QA chain — the core RAG generation step.

`build_qa_chain()` returns a callable that accepts a question + retrieved chunks
and produces a grounded answer.  It is deliberately thin: prompt construction
and LLM invocation, nothing more.  The answer_service.py orchestrates the full
retrieve → answer → cite flow.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.llm.factory import get_llm
from src.llm.prompts import (
    NO_CONTEXT_RESPONSE,
    build_prompt_messages,
    format_context_blocks,
)
from src.utils.logging import get_logger

log = get_logger(__name__)

# Retry only on transient network / rate-limit errors from OpenAI.
# ValueError, TypeError, AttributeError etc. are programming errors — retrying them
# wastes API quota and hides bugs.
try:
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    _TRANSIENT_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError)
except ImportError:  # openai not installed (e.g. using Gemini only)
    _TRANSIENT_ERRORS = (OSError,)


def build_qa_chain(llm: Optional[BaseChatModel] = None) -> Callable:
    """
    Return a callable QA chain.

    Args:
        llm: Optional pre-constructed LLM.  Uses get_llm() if not provided.

    Returns:
        A function with signature:
            chain(question: str, chunks: List[RetrievedChunk]) -> str
    """
    model = llm or get_llm()

    @retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _invoke_with_retry(messages: List[BaseMessage]) -> str:
        response = model.invoke(messages)
        return response.content

    def chain(question: str, chunks) -> str:
        """
        Generate a grounded answer for *question* from *chunks*.

        Args:
            question: User's question string.
            chunks:   List of RetrievedChunk objects.

        Returns:
            The model's answer as a string.
        """
        if not chunks:
            log.info("No chunks retrieved — returning no-context response.")
            return NO_CONTEXT_RESPONSE

        context_str = format_context_blocks(chunks)
        messages = build_prompt_messages(question, context_str)

        log.info(
            "Invoking LLM for question: '%s' with %d context chunks.",
            question[:80],
            len(chunks),
        )

        try:
            answer = _invoke_with_retry(messages)
        except Exception as exc:
            log.error("LLM invocation failed: %s", exc)
            raise RuntimeError(
                f"The language model failed to respond: {exc}"
            ) from exc

        log.info("LLM responded (%d chars).", len(answer))
        return answer

    return chain
