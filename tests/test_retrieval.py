"""
Tests for retrieval logic and the VectorStoreRetriever.

Most of these tests mock the underlying vector store to avoid requiring
a live ChromaDB instance and API keys in CI.  Integration tests that require
real infrastructure are marked with @pytest.mark.integration.

Validates:
  - retriever returns RetrievedChunk objects
  - threshold filtering works correctly
  - empty retrieval produces empty list
  - error from vector store is wrapped in RuntimeError
  - count_documents delegates to store
  - get_vector_store raises ValueError for unknown backend
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.vectordb.retriever import RetrievedChunk, VectorStoreRetriever


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_doc(content: str, filename: str = "algo.pdf", page: int = 1) -> Document:
    return Document(
        page_content=content,
        metadata={"filename": filename, "page": page, "chunk_id": "abc", "char_count": len(content)},
    )


def _mock_store(results: list[tuple[Document, float]]) -> MagicMock:
    """Return a mock vector store that returns *results* from similarity_search_with_score."""
    store = MagicMock()
    store.similarity_search_with_score.return_value = results
    store.count.return_value = len(results)
    return store


# ── RetrievedChunk ────────────────────────────────────────────────────────────


class TestRetrievedChunk:

    def test_content_property(self):
        doc = _make_doc("BFS uses a queue.")
        chunk = RetrievedChunk(document=doc, score=0.9)
        assert chunk.content == "BFS uses a queue."

    def test_metadata_property(self):
        doc = _make_doc("content", filename="algo.pdf")
        chunk = RetrievedChunk(document=doc, score=0.8)
        assert chunk.metadata["filename"] == "algo.pdf"

    def test_score_stored(self):
        doc = _make_doc("content")
        chunk = RetrievedChunk(document=doc, score=0.753)
        assert chunk.score == pytest.approx(0.753)


# ── VectorStoreRetriever ──────────────────────────────────────────────────────


class TestVectorStoreRetriever:

    def test_retrieve_returns_chunks(self):
        raw = [
            (_make_doc("BFS explanation"), 0.95),
            (_make_doc("DFS explanation"), 0.80),
        ]
        store = _mock_store(raw)
        retriever = VectorStoreRetriever(store=store, top_k=4, similarity_threshold=0.0)
        chunks = retriever.retrieve("What is BFS?")
        assert len(chunks) == 2
        assert all(isinstance(c, RetrievedChunk) for c in chunks)

    def test_retrieve_passes_top_k_to_store(self):
        """VectorStoreRetriever must forward top_k to the underlying store call."""
        store = MagicMock()
        store.similarity_search_with_score.return_value = []
        retriever = VectorStoreRetriever(store=store, top_k=5, similarity_threshold=0.0)

        retriever.retrieve("query", top_k=3)

        store.similarity_search_with_score.assert_called_once_with("query", k=3)

    def test_retrieve_uses_instance_top_k_when_not_overridden(self):
        store = MagicMock()
        store.similarity_search_with_score.return_value = []
        retriever = VectorStoreRetriever(store=store, top_k=7, similarity_threshold=0.0)

        retriever.retrieve("query")

        store.similarity_search_with_score.assert_called_once_with("query", k=7)

    def test_threshold_filters_low_score_chunks(self):
        raw = [
            (_make_doc("relevant"), 0.85),
            (_make_doc("irrelevant"), 0.20),
        ]
        store = _mock_store(raw)
        retriever = VectorStoreRetriever(store=store, top_k=4, similarity_threshold=0.5)
        chunks = retriever.retrieve("query")
        assert len(chunks) == 1
        assert chunks[0].content == "relevant"

    def test_zero_threshold_returns_all(self):
        raw = [
            (_make_doc("a"), 0.1),
            (_make_doc("b"), 0.05),
        ]
        store = _mock_store(raw)
        retriever = VectorStoreRetriever(store=store, top_k=4, similarity_threshold=0.0)
        chunks = retriever.retrieve("query")
        assert len(chunks) == 2

    def test_empty_retrieval_returns_empty_list(self):
        store = _mock_store([])
        retriever = VectorStoreRetriever(store=store, top_k=4, similarity_threshold=0.0)
        chunks = retriever.retrieve("obscure question about nothing")
        assert chunks == []

    def test_store_error_raises_runtime_error(self):
        store = MagicMock()
        store.similarity_search_with_score.side_effect = Exception("DB connection failed")
        retriever = VectorStoreRetriever(store=store, top_k=4, similarity_threshold=0.0)
        with pytest.raises(RuntimeError, match="Retrieval error"):
            retriever.retrieve("query")

    def test_count_documents_delegates_to_store(self):
        store = MagicMock()
        store.count.return_value = 42
        retriever = VectorStoreRetriever(store=store, top_k=4, similarity_threshold=0.0)
        assert retriever.count_documents() == 42

    def test_count_documents_returns_minus1_on_error(self):
        store = MagicMock()
        store.count.side_effect = Exception("error")
        retriever = VectorStoreRetriever(store=store, top_k=4, similarity_threshold=0.0)
        assert retriever.count_documents() == -1

    def test_set_threshold_updates_filtering(self):
        """set_threshold() should be reflected in subsequent retrieve() calls."""
        raw = [
            (_make_doc("high"), 0.90),
            (_make_doc("low"), 0.20),
        ]
        store = _mock_store(raw)
        retriever = VectorStoreRetriever(store=store, top_k=4, similarity_threshold=0.0)

        # No threshold — both returned
        assert len(retriever.retrieve("q")) == 2

        # Raise threshold via public API
        retriever.set_threshold(0.5)
        assert len(retriever.retrieve("q")) == 1


# ── get_vector_store factory ──────────────────────────────────────────────────


class TestVectorStoreFactory:

    def test_invalid_backend_raises_value_error(self):
        from src.vectordb.retriever import get_vector_store

        with patch("src.vectordb.retriever.get_settings") as mock_settings, \
             patch("src.vectordb.retriever.get_embedding_model"):
            mock_settings.return_value.vector_db = "postgres"
            with pytest.raises(ValueError, match="Unsupported VECTOR_DB"):
                get_vector_store()
