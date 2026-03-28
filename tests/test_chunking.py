"""
Tests for the chunking pipeline.

Validates:
  - chunks are produced and non-empty
  - chunk sizes stay within configured bounds
  - metadata is propagated to every chunk
  - chunk_id and chunk_index fields are set
  - code-heavy flag is respected (different splitter path)
  - custom chunk size / overlap overrides work
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents
from src.ingestion.parser import parse_documents


class TestChunking:

    def test_produces_chunks(self, sample_document):
        chunks = chunk_documents([sample_document])
        assert len(chunks) > 0, "Should produce at least one chunk"

    def test_no_empty_chunks(self, sample_document):
        chunks = chunk_documents([sample_document])
        for chunk in chunks:
            assert chunk.page_content.strip(), "Every chunk must have non-empty content"

    def test_chunk_size_respected(self, sample_document):
        """Chunks should not wildly exceed the configured size."""
        chunk_size = 300
        chunks = chunk_documents([sample_document], chunk_size=chunk_size, chunk_overlap=50)
        for chunk in chunks:
            # Allow a small buffer for separator inclusion
            assert len(chunk.page_content) <= chunk_size * 1.5, (
                f"Chunk too large: {len(chunk.page_content)} chars (limit ~{chunk_size * 1.5})"
            )

    def test_metadata_propagated(self, sample_document):
        """Core metadata fields must survive into every chunk."""
        chunks = chunk_documents([sample_document])
        for chunk in chunks:
            assert "filename" in chunk.metadata
            assert "page" in chunk.metadata
            assert "source" in chunk.metadata
            assert chunk.metadata["filename"] == "algorithms.pdf"
            assert chunk.metadata["page"] == 1

    def test_chunk_id_set(self, sample_document):
        chunks = chunk_documents([sample_document])
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert len(chunk.metadata["chunk_id"]) == 16

    def test_chunk_index_set(self, sample_document):
        chunks = chunk_documents([sample_document])
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert isinstance(chunk.metadata["chunk_index"], int)

    def test_chunk_indices_monotonically_increase(self, sample_documents):
        chunks = chunk_documents(sample_documents)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == sorted(indices), "chunk_index values should be monotonically increasing"

    def test_char_count_matches_content(self, sample_document):
        chunks = chunk_documents([sample_document])
        for chunk in chunks:
            assert chunk.metadata["char_count"] == len(chunk.page_content)

    def test_multiple_documents(self, sample_documents):
        chunks = chunk_documents(sample_documents)
        filenames = {c.metadata["filename"] for c in chunks}
        assert "algorithms.pdf" in filenames
        assert "data_structures.pdf" in filenames

    def test_overlap_creates_shared_content(self, sample_document):
        """With overlap, consecutive chunks should share some content."""
        chunk_size = 200
        overlap = 80
        chunks = chunk_documents([sample_document], chunk_size=chunk_size, chunk_overlap=overlap)
        if len(chunks) < 2:
            pytest.skip("Document too short to produce overlapping chunks at this size")

        for i in range(len(chunks) - 1):
            end_of_first = chunks[i].page_content[-overlap:]
            start_of_second = chunks[i + 1].page_content[:overlap]
            # They don't need to be identical (separators differ), but there should be overlap
            shared = set(end_of_first.split()) & set(start_of_second.split())
            # Only assert if document is long enough to have meaningful overlap
            if len(chunks[i].page_content) >= chunk_size:
                assert len(shared) > 0, f"Expected some shared words between chunks {i} and {i+1}"

    def test_code_heavy_document_uses_code_splitter(self):
        """Documents flagged as code-heavy should still produce valid chunks."""
        code_doc = Document(
            page_content=(
                "    public int insert(Node root, int value) {\n"
                "        if (root == null) return new Node(value);\n"
                "        if (value < root.val) root.left = insert(root.left, value);\n"
                "        else root.right = insert(root.right, value);\n"
                "        return root;\n"
                "    }\n" * 20  # Repeat to create a document with lots of indented code
            ),
            metadata={
                "source": "/data/raw/code.txt",
                "filename": "code.txt",
                "page": 1,
                "total_pages": 1,
                "file_type": "text",
                "is_code_heavy": True,
                "section_heading": "",
            },
        )
        chunks = chunk_documents([code_doc])
        assert len(chunks) > 0

    def test_empty_document_list(self):
        chunks = chunk_documents([])
        assert chunks == []

    def test_custom_chunk_size_override(self, sample_document):
        """Custom chunk_size arg should override settings."""
        small_chunks = chunk_documents([sample_document], chunk_size=100, chunk_overlap=10)
        large_chunks = chunk_documents([sample_document], chunk_size=2000, chunk_overlap=100)
        assert len(small_chunks) >= len(large_chunks), (
            "Smaller chunk size should produce more chunks"
        )

    def test_zero_overlap_override_is_honoured(self, sample_document):
        no_overlap_chunks = chunk_documents([sample_document], chunk_size=120, chunk_overlap=0)
        overlap_chunks = chunk_documents([sample_document], chunk_size=120, chunk_overlap=40)

        assert len(no_overlap_chunks) > 1
        assert len(overlap_chunks) >= len(no_overlap_chunks)

        for i in range(len(no_overlap_chunks) - 1):
            trailing_words = set(no_overlap_chunks[i].page_content.split()[-8:])
            leading_words = set(no_overlap_chunks[i + 1].page_content.split()[:8])
            assert len(trailing_words & leading_words) <= 1

    def test_invalid_chunk_size_raises(self, sample_document):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_documents([sample_document], chunk_size=0, chunk_overlap=0)

    def test_invalid_chunk_overlap_raises(self, sample_document):
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_documents([sample_document], chunk_size=100, chunk_overlap=100)

    def test_negative_chunk_overlap_raises(self, sample_document):
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            chunk_documents([sample_document], chunk_size=100, chunk_overlap=-1)
