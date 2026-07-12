"""
Tests for the AnswerService orchestration layer.
"""

from __future__ import annotations

import pytest

from src.services.answer_service import AnswerService


class DummyRetriever:
    def __init__(self, count: int = 1) -> None:
        self._count = count
        self.threshold = None
        self.last_top_k = None
        self.last_source_filter = None
        self.retrieve_calls = 0

    def count_documents(self) -> int:
        return self._count

    def retrieve(self, question: str, top_k=None, source_filter=None):
        self.retrieve_calls += 1
        self.last_top_k = top_k
        self.last_source_filter = source_filter
        return []

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold


class TestAnswerService:
    def test_configure_retrieval_updates_top_k_and_threshold(self):
        retriever = DummyRetriever()
        service = AnswerService(
            retriever=retriever,
            qa_chain=lambda question, chunks: "ok",
            top_k=4,
        )

        service.configure_retrieval(top_k=6, threshold=0.45)
        service.answer("What is BFS?")

        assert retriever.threshold == 0.45
        assert retriever.last_top_k == 6

    def test_configure_retrieval_rejects_invalid_top_k(self):
        service = AnswerService(
            retriever=DummyRetriever(),
            qa_chain=lambda question, chunks: "ok",
            top_k=4,
        )

        with pytest.raises(ValueError, match="top_k must be positive"):
            service.configure_retrieval(top_k=0)

    def test_answer_rejects_invalid_top_k(self):
        retriever = DummyRetriever()
        qa_calls = []
        service = AnswerService(
            retriever=retriever,
            qa_chain=lambda question, chunks: qa_calls.append((question, chunks)) or "ok",
            top_k=4,
        )

        result = service.answer("What is BFS?", top_k=0)

        assert result.is_error is True
        assert result.error == "Invalid top_k."
        assert "top_k must be a positive integer" in result.answer
        assert retriever.retrieve_calls == 0
        assert qa_calls == []

    def test_answer_study_passes_mode_and_source_filter(self):
        retriever = DummyRetriever()
        calls = []
        service = AnswerService(
            retriever=retriever,
            qa_chain=lambda question, chunks, study_mode="answer": calls.append(study_mode) or "ok",
            top_k=4,
        )

        result = service.answer_study(
            "Make flashcards",
            study_mode="flashcards",
            source_filter="calculus.pdf",
        )

        assert result.answer == "ok"
        assert calls == ["flashcards"]
        assert retriever.last_source_filter == "calculus.pdf"
