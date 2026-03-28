"""
Retrieval evaluation metrics.

Implements lightweight, dependency-free metrics suitable for evaluating a RAG
retrieval pipeline without requiring an external evaluation framework.

Metrics:
  - hit@k:           Was any expected source document retrieved in top-k results?
  - precision@k:     Fraction of retrieved chunks whose source is in the expected set.
  - mrr@k:           Mean Reciprocal Rank — position of first relevant result.
  - context_recall:  How many expected source docs appear in the retrieved set?
  - avg_score:       Average similarity score of retrieved chunks.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalCase:
    """A single evaluation test case."""
    question: str
    expected_sources: List[str]  # List of filenames that should appear in retrieved context
    expected_keywords: List[str] = field(default_factory=list)  # Keywords that should appear in answer
    description: str = ""


@dataclass
class EvalResult:
    """Results for a single evaluation case."""
    case: EvalCase
    retrieved_sources: List[str]
    retrieved_scores: List[float]
    answer: str
    hit_at_k: bool
    precision_at_k: float
    reciprocal_rank: float
    context_recall: float
    avg_score: float
    keyword_coverage: float
    error: Optional[str] = None


class RetrievalMetrics:
    """Compute retrieval quality metrics."""

    @staticmethod
    def hit_at_k(retrieved_sources: List[str], expected_sources: List[str]) -> bool:
        """Return True if at least one expected source appears in the retrieved set."""
        expected_set = {s.lower() for s in expected_sources}
        retrieved_set = {s.lower() for s in retrieved_sources}
        return bool(expected_set & retrieved_set)

    @staticmethod
    def precision_at_k(retrieved_sources: List[str], expected_sources: List[str]) -> float:
        """Fraction of retrieved chunks whose source is in the expected set."""
        if not retrieved_sources:
            return 0.0
        expected_set = {s.lower() for s in expected_sources}
        relevant = sum(1 for s in retrieved_sources if s.lower() in expected_set)
        return relevant / len(retrieved_sources)

    @staticmethod
    def reciprocal_rank(retrieved_sources: List[str], expected_sources: List[str]) -> float:
        """
        Return 1/rank of the first relevant result, or 0.0 if none found.
        Rank is 1-indexed.
        """
        expected_set = {s.lower() for s in expected_sources}
        for rank, src in enumerate(retrieved_sources, start=1):
            if src.lower() in expected_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def context_recall(retrieved_sources: List[str], expected_sources: List[str]) -> float:
        """Fraction of expected sources that appear anywhere in the retrieved set."""
        if not expected_sources:
            return 1.0
        expected_set = {s.lower() for s in expected_sources}
        retrieved_set = {s.lower() for s in retrieved_sources}
        found = expected_set & retrieved_set
        return len(found) / len(expected_set)

    @staticmethod
    def keyword_coverage(answer: str, keywords: List[str]) -> float:
        """Fraction of expected keywords that appear (case-insensitive) in the answer."""
        if not keywords:
            return 1.0
        answer_lower = answer.lower()
        found = sum(1 for kw in keywords if kw.lower() in answer_lower)
        return found / len(keywords)

    @staticmethod
    def avg_score(scores: List[float]) -> float:
        """Return mean of a list of similarity scores."""
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @classmethod
    def compute(
        cls,
        case: EvalCase,
        retrieved_sources: List[str],
        retrieved_scores: List[float],
        answer: str,
        error: Optional[str] = None,
    ) -> EvalResult:
        """Compute all metrics for a single eval case."""
        return EvalResult(
            case=case,
            retrieved_sources=retrieved_sources,
            retrieved_scores=retrieved_scores,
            answer=answer,
            hit_at_k=cls.hit_at_k(retrieved_sources, case.expected_sources),
            precision_at_k=cls.precision_at_k(retrieved_sources, case.expected_sources),
            reciprocal_rank=cls.reciprocal_rank(retrieved_sources, case.expected_sources),
            context_recall=cls.context_recall(retrieved_sources, case.expected_sources),
            avg_score=cls.avg_score(retrieved_scores),
            keyword_coverage=cls.keyword_coverage(answer, case.expected_keywords),
            error=error,
        )
