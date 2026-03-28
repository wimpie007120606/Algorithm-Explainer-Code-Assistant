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

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.llm.qa_chain import build_qa_chain
from src.vectordb.retriever import RetrievedChunk, VectorStoreRetriever
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class AnswerResult:
    """Container for a complete RAG response."""

    question: str
    answer: str
    chunks: List[RetrievedChunk]
    has_context: bool
    error: Optional[str] = None

    # Derived convenience fields
    source_filenames: List[str] = field(default_factory=list)

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
        retriever: Optional[VectorStoreRetriever] = None,
        qa_chain: Optional[Callable] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self._retriever = retriever or VectorStoreRetriever()
        self._chain = qa_chain or build_qa_chain()
        self._top_k = top_k

    # ── public API ────────────────────────────────────────────────────────────

    def answer(self, question: str, top_k: Optional[int] = None) -> AnswerResult:
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
        question = question.strip()
        if not question:
            return AnswerResult(
                question=question,
                answer="Please enter a question.",
                chunks=[],
                has_context=False,
                error="Empty question.",
            )

        k = top_k or self._top_k

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
            chunks = self._retriever.retrieve(question, top_k=k)
        except RuntimeError as exc:
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
