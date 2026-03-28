"""
Evaluation runner.

Executes a suite of EvalCases against the live RAG pipeline and produces a
structured report.  Designed to be run via `python scripts/run_eval.py`.

The default eval suite tests common algorithm questions.  Replace or extend
the `DEFAULT_EVAL_CASES` list with cases relevant to your ingested documents.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.evaluation.metrics import EvalCase, EvalResult, RetrievalMetrics
from src.services.answer_service import AnswerService
from src.vectordb.retriever import VectorStoreRetriever
from src.utils.logging import get_logger

log = get_logger(__name__)

# ─── Default eval suite ───────────────────────────────────────────────────────
# These cases target the included sample document (data/raw/sample_algorithms.md).
# expected_sources must match the exact filename (not full path) of the ingested file.
# Update these when you ingest your own PDFs — set expected_sources to the
# filename(s) that should contain the answer.

DEFAULT_EVAL_CASES: List[EvalCase] = [
    EvalCase(
        question="What is the time complexity of Dijkstra's algorithm?",
        expected_sources=["sample_algorithms.md"],
        expected_keywords=["dijkstra", "O("],
        description="Complexity query — Dijkstra",
    ),
    EvalCase(
        question="Explain the Max-Flow Min-Cut theorem.",
        expected_sources=["sample_algorithms.md"],
        expected_keywords=["max-flow", "min-cut"],
        description="Theoretical concept query",
    ),
    EvalCase(
        question="What is a quadtree and how does insertion work?",
        expected_sources=["sample_algorithms.md"],
        expected_keywords=["quadtree", "insert"],
        description="Data structure + operation query",
    ),
    EvalCase(
        question="Compare BFS and DFS graph traversal.",
        expected_sources=["sample_algorithms.md"],
        expected_keywords=["BFS", "DFS"],
        description="Comparison query",
    ),
]


@dataclass
class EvalReport:
    """Aggregated evaluation report."""

    results: List[EvalResult]
    total_cases: int
    cases_with_error: int
    hit_rate: float          # fraction of cases with hit@k = True
    mean_precision: float
    mean_mrr: float
    mean_recall: float
    mean_keyword_coverage: float
    mean_avg_score: float
    elapsed_seconds: float

    def print_report(self) -> None:
        """Print a human-readable report to stdout."""
        separator = "─" * 60

        print(f"\n{separator}")
        print("RETRIEVAL EVALUATION REPORT")
        print(separator)
        print(f"  Total cases       : {self.total_cases}")
        print(f"  Cases with errors : {self.cases_with_error}")
        print(f"  Hit Rate (hit@k)  : {self.hit_rate:.1%}")
        print(f"  Mean Precision@k  : {self.mean_precision:.1%}")
        print(f"  Mean MRR          : {self.mean_mrr:.3f}")
        print(f"  Mean Recall       : {self.mean_recall:.1%}")
        print(f"  Mean Kw Coverage  : {self.mean_keyword_coverage:.1%}")
        print(f"  Mean Sim Score    : {self.mean_avg_score:.3f}")
        print(f"  Elapsed (s)       : {self.elapsed_seconds:.1f}")
        print(separator)
        print("\nPer-Case Results:")
        for i, r in enumerate(self.results, start=1):
            status = "ERROR" if r.error else ("HIT" if r.hit_at_k else "MISS")
            print(f"\n  [{i}] {r.case.description or r.case.question[:60]}")
            print(f"       Status       : {status}")
            print(f"       Precision@k  : {r.precision_at_k:.1%}")
            print(f"       MRR          : {r.reciprocal_rank:.3f}")
            print(f"       Context Rec  : {r.context_recall:.1%}")
            print(f"       Kw Coverage  : {r.keyword_coverage:.1%}")
            print(f"       Avg Score    : {r.avg_score:.3f}")
            if r.retrieved_sources:
                print(f"       Sources      : {', '.join(set(r.retrieved_sources))}")
            if r.error:
                print(f"       Error        : {r.error}")
        print(f"\n{separator}\n")

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "cases_with_error": self.cases_with_error,
            "hit_rate": round(self.hit_rate, 4),
            "mean_precision": round(self.mean_precision, 4),
            "mean_mrr": round(self.mean_mrr, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_keyword_coverage": round(self.mean_keyword_coverage, 4),
            "mean_avg_score": round(self.mean_avg_score, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


class EvalRunner:
    """
    Runs an eval suite and returns an EvalReport.

    Args:
        cases:    List of EvalCase objects.  Defaults to DEFAULT_EVAL_CASES.
        top_k:    Number of chunks to retrieve per question.
        dry_run:  If True, only run retrieval (no LLM call).  Faster and cheaper.
    """

    def __init__(
        self,
        cases: Optional[List[EvalCase]] = None,
        top_k: int = 4,
        dry_run: bool = False,
    ) -> None:
        self._cases = cases or DEFAULT_EVAL_CASES
        self._top_k = top_k
        self._dry_run = dry_run
        self._retriever = VectorStoreRetriever(top_k=top_k)
        self._service = AnswerService(retriever=self._retriever, top_k=top_k)

    def run(self) -> EvalReport:
        """Execute all eval cases and return an EvalReport."""
        log.info("Starting evaluation: %d cases, top_k=%d, dry_run=%s", len(self._cases), self._top_k, self._dry_run)
        start = time.time()

        results: List[EvalResult] = []

        for i, case in enumerate(self._cases, start=1):
            log.info("[%d/%d] Evaluating: %s", i, len(self._cases), case.question[:60])
            result = self._eval_case(case)
            results.append(result)

        elapsed = time.time() - start

        # Aggregate
        def _mean(values: List[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        hit_rate = _mean([float(r.hit_at_k) for r in results])
        mean_precision = _mean([r.precision_at_k for r in results])
        mean_mrr = _mean([r.reciprocal_rank for r in results])
        mean_recall = _mean([r.context_recall for r in results])
        mean_kw = _mean([r.keyword_coverage for r in results])
        mean_score = _mean([r.avg_score for r in results])
        errors = sum(1 for r in results if r.error is not None)

        report = EvalReport(
            results=results,
            total_cases=len(results),
            cases_with_error=errors,
            hit_rate=hit_rate,
            mean_precision=mean_precision,
            mean_mrr=mean_mrr,
            mean_recall=mean_recall,
            mean_keyword_coverage=mean_kw,
            mean_avg_score=mean_score,
            elapsed_seconds=elapsed,
        )
        return report

    def _eval_case(self, case: EvalCase) -> EvalResult:
        try:
            chunks = self._retriever.retrieve(case.question, top_k=self._top_k)
            retrieved_sources = [c.metadata.get("filename", "unknown") for c in chunks]
            retrieved_scores = [c.score for c in chunks]

            if self._dry_run:
                answer = "[dry-run — LLM not called]"
            else:
                ar = self._service.answer(case.question, top_k=self._top_k)
                answer = ar.answer

            return RetrievalMetrics.compute(
                case=case,
                retrieved_sources=retrieved_sources,
                retrieved_scores=retrieved_scores,
                answer=answer,
            )

        except Exception as exc:
            log.error("Eval case failed: %s — %s", case.question[:60], exc)
            return RetrievalMetrics.compute(
                case=case,
                retrieved_sources=[],
                retrieved_scores=[],
                answer="",
                error=str(exc),
            )
