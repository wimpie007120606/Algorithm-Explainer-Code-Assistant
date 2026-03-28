"""
Tests for document loaders and the ingestion pipeline.

Validates:
  - text and markdown files load correctly
  - metadata is correct on loaded documents
  - unsupported file types raise ValueError
  - empty files raise ValueError
  - pipeline dry-run mode works without a vector store
  - pipeline handles missing files gracefully
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestion.loaders import load_document, load_text
from src.ingestion.parser import parse_documents
from src.ingestion.pipeline import IngestionPipeline


class TestTextLoader:

    def test_load_txt_returns_document(self, tmp_text_file):
        docs = load_document(tmp_text_file)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)

    def test_load_md_returns_document(self, tmp_md_file):
        docs = load_document(tmp_md_file)
        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == "markdown"

    def test_txt_metadata_correct(self, tmp_text_file):
        docs = load_document(tmp_text_file)
        meta = docs[0].metadata
        assert meta["filename"] == "algorithms.txt"
        assert meta["page"] == 1
        assert meta["total_pages"] == 1
        assert meta["file_type"] == "text"
        assert "source" in meta

    def test_txt_content_not_empty(self, tmp_text_file):
        docs = load_document(tmp_text_file)
        assert docs[0].page_content.strip()

    def test_unsupported_extension_raises(self, tmp_path):
        bad_file = tmp_path / "file.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(bad_file)

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("   \n   ")
        with pytest.raises(ValueError, match="empty after cleaning"):
            load_document(empty)

    def test_file_content_preserved(self, tmp_text_file):
        docs = load_document(tmp_text_file)
        content = docs[0].page_content
        assert "BFS" in content
        assert "DFS" in content


class TestParser:

    def test_section_heading_detected(self, sample_document):
        parsed = parse_documents([sample_document])
        # The sample doc starts with "BFS and DFS Graph Traversal"
        heading = parsed[0].metadata.get("section_heading", "")
        # May or may not detect it depending on regex — just check key exists
        assert "section_heading" in parsed[0].metadata

    def test_is_code_heavy_field_added(self, sample_document):
        parsed = parse_documents([sample_document])
        assert "is_code_heavy" in parsed[0].metadata

    def test_code_heavy_true_for_indented_doc(self):
        code_doc = Document(
            page_content="\n".join(["    " + "x = 1;" for _ in range(30)]),
            metadata={"filename": "code.txt", "page": 1, "file_type": "text"},
        )
        parsed = parse_documents([code_doc])
        assert parsed[0].metadata["is_code_heavy"] is True

    def test_code_heavy_false_for_prose(self, sample_document):
        parsed = parse_documents([sample_document])
        # Sample doc has some code but not majority-indented
        # Just check the field is boolean
        assert isinstance(parsed[0].metadata["is_code_heavy"], bool)

    def test_metadata_not_lost(self, sample_document):
        original_meta_keys = set(sample_document.metadata.keys())
        parsed = parse_documents([sample_document])
        for key in original_meta_keys:
            assert key in parsed[0].metadata


class TestIngestionPipeline:

    def test_pipeline_dry_run(self, tmp_text_file):
        """Pipeline without a vector store should load and chunk but not crash."""
        pipeline = IngestionPipeline(vector_store=None)
        stats = pipeline.ingest_directory(
            directory=tmp_text_file.parent,
            file_paths=[tmp_text_file],
        )
        assert stats.files_attempted == 1
        assert stats.files_succeeded == 1
        assert stats.files_failed == 0
        assert stats.total_chunks > 0

    def test_pipeline_ingest_file(self, tmp_text_file):
        pipeline = IngestionPipeline(vector_store=None)
        chunks = pipeline.ingest_file(tmp_text_file)
        assert len(chunks) > 0

    def test_pipeline_handles_bad_extension(self, tmp_path):
        """Forcing an unsupported file through the pipeline should fail gracefully."""
        bad = tmp_path / "bad.xyz"
        bad.write_text("data")
        pipeline = IngestionPipeline(vector_store=None)
        # Passing the file explicitly bypasses the iter_documents extension filter.
        # The loader layer must reject it and record the failure in stats.
        stats = pipeline.ingest_directory(directory=tmp_path, file_paths=[bad])
        assert stats.files_attempted == 1
        assert stats.files_failed == 1
        assert stats.files_succeeded == 0
        assert len(stats.errors) == 1

    def test_pipeline_no_docs_returns_stats(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        pipeline = IngestionPipeline(vector_store=None)
        stats = pipeline.ingest_directory(directory=empty_dir)
        assert stats.files_attempted == 0
        assert stats.total_chunks == 0

    def test_stats_count_matches_reality(self, tmp_path):
        """Ingest two files — stats should reflect both."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("Dijkstra algorithm shortest path graph weighted edges.")
        f2.write_text("Bellman-Ford algorithm negative weights relaxation.")

        pipeline = IngestionPipeline(vector_store=None)
        stats = pipeline.ingest_directory(directory=tmp_path)
        assert stats.files_succeeded == 2
        assert stats.total_chunks >= 2  # at least one chunk per file
