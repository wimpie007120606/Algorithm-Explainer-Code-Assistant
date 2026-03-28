"""
Tests for the prompt construction pipeline.

Validates:
  - system prompt contains the required grounding instructions
  - context blocks are correctly formatted
  - build_prompt_messages returns correct message structure
  - no-context response is triggered when chunks are empty
  - QA chain returns NO_CONTEXT_RESPONSE on empty chunks
"""

from __future__ import annotations

import pytest

from src.llm.prompts import (
    SYSTEM_PROMPT,
    NO_CONTEXT_RESPONSE,
    CONTEXT_BLOCK_TEMPLATE,
    format_context_blocks,
    build_prompt_messages,
)
from src.vectordb.retriever import RetrievedChunk
from langchain_core.documents import Document


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk(content: str, filename: str = "test.pdf", page: int = 1, score: float = 0.85) -> RetrievedChunk:
    doc = Document(
        page_content=content,
        metadata={"filename": filename, "page": page, "chunk_id": "abc123", "char_count": len(content)},
    )
    return RetrievedChunk(document=doc, score=score)


# ── System prompt enforcement ─────────────────────────────────────────────────

class TestSystemPrompt:

    def test_system_prompt_not_empty(self):
        assert SYSTEM_PROMPT.strip()

    def test_prohibits_fabrication(self):
        """The system prompt must explicitly forbid inventing information."""
        prompt_lower = SYSTEM_PROMPT.lower()
        # At least one of these grounding keywords must appear
        grounding_keywords = ["only", "retrieved context", "do not invent", "must not", "not contain"]
        assert any(kw in prompt_lower for kw in grounding_keywords), (
            "System prompt must contain explicit grounding/prohibition instructions"
        )

    def test_contains_insufficient_information_phrase(self):
        """The system prompt must instruct the model to say when info is missing."""
        assert "not contain enough information" in SYSTEM_PROMPT.lower() or \
               "do not have enough" in SYSTEM_PROMPT.lower() or \
               "insufficient" in SYSTEM_PROMPT.lower(), \
               "System prompt must instruct model to declare when context is insufficient"

    def test_mentions_java_code(self):
        """The prompt should address Java code specifically."""
        assert "java" in SYSTEM_PROMPT.lower(), "System prompt should reference Java code grounding"

    def test_answer_structure_defined(self):
        """The prompt should specify an answer structure."""
        assert "direct answer" in SYSTEM_PROMPT.lower() or \
               "answer format" in SYSTEM_PROMPT.lower(), \
               "System prompt should define an expected answer format"


# ── Context formatting ────────────────────────────────────────────────────────

class TestContextFormatting:

    def test_format_context_with_chunks(self):
        chunks = [
            _make_chunk("BFS uses a queue.", "graphs.pdf", page=2, score=0.9),
            _make_chunk("DFS uses a stack.", "graphs.pdf", page=3, score=0.7),
        ]
        context = format_context_blocks(chunks)
        assert "graphs.pdf" in context
        assert "BFS uses a queue." in context
        assert "DFS uses a stack." in context

    def test_format_context_includes_page(self):
        chunks = [_make_chunk("content", "algo.pdf", page=5)]
        context = format_context_blocks(chunks)
        assert "Page 5" in context

    def test_format_context_numbers_sources(self):
        chunks = [
            _make_chunk("chunk one", score=0.9),
            _make_chunk("chunk two", score=0.8),
        ]
        context = format_context_blocks(chunks)
        assert "SOURCE 1" in context
        assert "SOURCE 2" in context

    def test_format_context_empty_chunks(self):
        context = format_context_blocks([])
        assert context  # Should not be empty — should have a placeholder message
        assert "No relevant context" in context

    def test_format_context_preserves_content(self):
        content = "Ford-Fulkerson finds maximum flow by augmenting paths repeatedly."
        chunks = [_make_chunk(content)]
        context = format_context_blocks(chunks)
        assert content in context


# ── build_prompt_messages ─────────────────────────────────────────────────────

class TestBuildPromptMessages:

    def test_returns_list_of_messages(self):
        msgs = build_prompt_messages("What is BFS?", "BFS uses a queue.")
        assert isinstance(msgs, list)
        assert len(msgs) == 2  # system + human

    def test_system_message_type(self):
        from langchain_core.messages import SystemMessage
        msgs = build_prompt_messages("Q", "context")
        assert isinstance(msgs[0], SystemMessage)

    def test_human_message_type(self):
        from langchain_core.messages import HumanMessage
        msgs = build_prompt_messages("Q", "context")
        assert isinstance(msgs[1], HumanMessage)

    def test_question_in_human_message(self):
        question = "How does Dijkstra work?"
        msgs = build_prompt_messages(question, "some context")
        human_text = msgs[1].content
        assert question in human_text

    def test_context_in_human_message(self):
        context = "Dijkstra uses a priority queue and relaxes edges."
        msgs = build_prompt_messages("Q", context)
        human_text = msgs[1].content
        assert context in human_text


# ── No-context fallback ───────────────────────────────────────────────────────

class TestNoContextResponse:

    def test_no_context_response_not_empty(self):
        assert NO_CONTEXT_RESPONSE.strip()

    def test_no_context_response_explains_situation(self):
        lower = NO_CONTEXT_RESPONSE.lower()
        # Must mention that no relevant content was found
        assert any(kw in lower for kw in ["not contain", "not enough", "no relevant", "knowledge base"])

    def test_qa_chain_returns_no_context_on_empty_chunks(self):
        """QA chain must return NO_CONTEXT_RESPONSE when given no chunks."""
        from src.llm.qa_chain import build_qa_chain
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        chain = build_qa_chain(llm=mock_llm)

        result = chain("What is BFS?", [])
        assert result == NO_CONTEXT_RESPONSE
        mock_llm.invoke.assert_not_called()
