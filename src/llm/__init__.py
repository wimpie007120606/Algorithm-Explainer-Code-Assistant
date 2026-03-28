"""LLM package — factory, prompts, and QA chain."""

from .factory import get_llm
from .qa_chain import build_qa_chain

__all__ = ["get_llm", "build_qa_chain"]
