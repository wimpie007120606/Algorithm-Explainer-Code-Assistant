"""
AnswerService — the main RAG orchestrator.

This is the primary entry point for the Streamlit UI and scripts.  It:
  1. Validates the knowledge base is not empty.
  2. Retrieves relevant chunks via VectorStoreRetriever.
  3. Invokes the QA chain to get a grounded answer.
  4. Packages the answer, retrieved chunks, and metadata into an AnswerResult.

All error paths produce clear, user-friendly messages rather than raw exceptions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from src.llm.qa_chain import build_qa_chain
from src.utils.logging import get_logger
from src.vectordb.retriever import RetrievedChunk, VectorStoreRetriever

log = get_logger(__name__)


@dataclass
class AnswerResult:
    """Container for a complete RAG response."""

    question: str
    answer: str
    chunks: list[RetrievedChunk]
    has_context: bool
    error: str | None = None

    # Derived convenience fields
    source_filenames: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.source_filenames:
            seen = set()
            for c in self.chunks:
                fn = c.metadata.get("filename", "Unknown")
                if fn not in seen:
                    self.source_filenames.append(fn)
                    seen.add(fn)

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)


class AnswerService:
    """
    Orchestrates the full RAG pipeline: retrieve → generate → package.

    Args:
        retriever: VectorStoreRetriever instance.
        qa_chain:  Callable chain(question, chunks) → str.
                   Defaults to the standard QA chain.
        top_k:     Number of chunks to retrieve (overrides retriever default).
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever | None = None,
        qa_chain: Callable | None = None,
        top_k: int | None = None,
    ) -> None:
        self._retriever = retriever or VectorStoreRetriever()
        self._chain = qa_chain or build_qa_chain()
        self._top_k = top_k

    # ── public API ────────────────────────────────────────────────────────────

    def configure_retrieval(
        self,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> None:
        """Update runtime retrieval settings without exposing internal fields."""
        if top_k is not None:
            if top_k <= 0:
                raise ValueError(f"top_k must be positive, got {top_k}.")
            self._top_k = top_k

        if threshold is not None:
            self._retriever.set_threshold(threshold)

    def answer(self, question: str, top_k: int | None = None) -> AnswerResult:
        """
        Answer *question* using retrieved context.

        Args:
            question: The user's natural-language question.
            top_k:    Number of chunks to retrieve (overrides instance default).

        Returns:
            AnswerResult with answer text, supporting chunks, and metadata.
            On error, AnswerResult.error is set and AnswerResult.answer contains
            a human-readable error message.
        """
        return self.answer_study(
            question=question,
            top_k=top_k,
            study_mode="answer",
            source_filter=None,
        )

    def answer_study(
        self,
        question: str,
        top_k: int | None = None,
        *,
        study_mode: str = "answer",
        source_filter: str | None = None,
    ) -> AnswerResult:
        """
        Answer *question* using retrieved context and a requested study mode.

        Args:
            question: User's natural-language request.
            top_k: Number of chunks to retrieve (overrides instance default).
            study_mode: Output style, e.g. answer, explain, summary, practice.
            source_filter: Optional filename to restrict retrieval.
        """
        question = question.strip()
        if not question:
            return AnswerResult(
                question=question,
                answer="Please enter a question.",
                chunks=[],
                has_context=False,
                error="Empty question.",
            )

        k = top_k if top_k is not None else self._top_k
        if k is not None and k <= 0:
            return AnswerResult(
                question=question,
                answer="Retrieval configuration error: top_k must be a positive integer.",
                chunks=[],
                has_context=False,
                error="Invalid top_k.",
            )

        # ── 1. Check knowledge base has content ──────────────────────────────
        doc_count = self._retriever.count_documents()
        if doc_count == 0:
            log.warning("Knowledge base is empty — no documents indexed.")
            return AnswerResult(
                question=question,
                answer=(
                    "The knowledge base is empty. Please ingest documents first using "
                    "the sidebar upload or by running `python scripts/ingest_docs.py`."
                ),
                chunks=[],
                has_context=False,
                error="Knowledge base empty.",
            )
        if doc_count < 0:
            log.error("Vector store returned error count (%d) — may be unavailable.", doc_count)
            return AnswerResult(
                question=question,
                answer=(
                    "Could not connect to the vector store. "
                    "Check your configuration and ensure ChromaDB is accessible."
                ),
                chunks=[],
                has_context=False,
                error="Vector store unavailable.",
            )

        # ── 2. Retrieve ───────────────────────────────────────────────────────
        try:
            chunks = self._retriever.retrieve(question, top_k=k, source_filter=source_filter)
        except (RuntimeError, ValueError) as exc:
            log.error("Retrieval failed: %s", exc)
            return AnswerResult(
                question=question,
                answer=f"Retrieval failed: {exc}",
                chunks=[],
                has_context=False,
                error=str(exc),
            )

        has_context = len(chunks) > 0

        # ── 3. Generate ───────────────────────────────────────────────────────
        try:
            try:
                answer = self._chain(question, chunks, study_mode=study_mode)
            except TypeError:
                answer = self._chain(question, chunks)
        except RuntimeError as exc:
            log.error("LLM generation failed: %s", exc)
            return AnswerResult(
                question=question,
                answer=f"The language model failed to respond: {exc}",
                chunks=chunks,
                has_context=has_context,
                error=str(exc),
            )

        return AnswerResult(
            question=question,
            answer=answer,
            chunks=chunks,
            has_context=has_context,
        )

    def is_ready(self) -> tuple[bool, str]:
        """
        Check whether the service is ready to answer questions.

        Returns:
            (True, "") if ready, or (False, "reason string") if not.
        """
        try:
            count = self._retriever.count_documents()
        except Exception as exc:
            return False, f"Vector store unavailable: {exc}"

        if count < 0:
            return False, "Vector store is unavailable (connection error)."
        if count == 0:
            return False, "No documents indexed. Please ingest documents first."

        return True, ""
