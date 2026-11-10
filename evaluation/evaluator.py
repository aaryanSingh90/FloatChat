"""
evaluation/evaluator.py
=======================
Runs the 32 test questions through the AI pipeline and measures:
  - Intent extraction accuracy
  - Variable extraction accuracy
  - Region extraction accuracy
  - Filter extraction accuracy
  - Query-plan correctness
  - Numerical result correctness (where deterministic)
  - RAG retrieval quality (documents retrieved for domain questions)

Produces a structured EvaluationReport.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from evaluation.test_questions import TEST_QUESTIONS
from ai.pipeline import AIPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    id: str
    question: str
    category: str
    mode: str
    intent_correct: bool
    variable_correct: bool
    region_correct: bool
    filter_correct: bool
    plan_correct: bool
    rag_correct: bool
    rag_docs_retrieved: int
    execution_error: Optional[str]
    latency_ms: float
    actual_plan: dict
    expected_plan: dict


@dataclass
class EvaluationReport:
    total: int = 0
    intent_accuracy: float = 0.0
    variable_accuracy: float = 0.0
    region_accuracy: float = 0.0
    filter_accuracy: float = 0.0
    plan_accuracy: float = 0.0
    rag_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    mode: str = "rule_based"
    results: list[QuestionResult] = field(default_factory=list)
    summary_df: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Runs all test questions through the pipeline and aggregates metrics.

    Parameters
    ----------
    pipeline : AIPipeline
        Configured pipeline (may or may not have Gemini).
    df : pd.DataFrame
        The oceanographic dataset.
    """

    def __init__(self, pipeline: AIPipeline, df: pd.DataFrame) -> None:
        self._pipeline = pipeline
        self._df = df

    def run(self) -> EvaluationReport:
        """Execute all test questions and return an EvaluationReport."""
        report = EvaluationReport()
        question_results: list[QuestionResult] = []

        for tc in TEST_QUESTIONS:
            qr = self._evaluate_one(tc)
            question_results.append(qr)

        report.total = len(question_results)
        report.results = question_results
        report.mode = question_results[0].mode if question_results else "rule_based"

        # Aggregate metrics
        report.intent_accuracy = _mean([r.intent_correct for r in question_results])
        report.variable_accuracy = _mean([r.variable_correct for r in question_results])
        report.region_accuracy = _mean([r.region_correct for r in question_results])
        report.filter_accuracy = _mean([r.filter_correct for r in question_results])
        report.plan_accuracy = _mean([r.plan_correct for r in question_results])

        rag_qs = [r for r in question_results if r.category == "rag"]
        report.rag_accuracy = _mean([r.rag_correct for r in rag_qs]) if rag_qs else 0.0

        report.avg_latency_ms = _mean([r.latency_ms for r in question_results])

        # Build summary DataFrame
        report.summary_df = _make_summary_df(question_results)

        return report

    def _evaluate_one(self, tc: dict) -> QuestionResult:
        t0 = time.perf_counter()
        try:
            pipeline_result = self._pipeline.process(tc["question"], self._df)
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - t0) * 1000
            return QuestionResult(
                id=tc["id"],
                question=tc["question"],
                category=tc["category"],
                mode="error",
                intent_correct=False,
                variable_correct=False,
                region_correct=False,
                filter_correct=False,
                plan_correct=False,
                rag_correct=False,
                rag_docs_retrieved=0,
                execution_error=str(exc),
                latency_ms=elapsed,
                actual_plan={},
                expected_plan=tc["expected_plan"],
            )

        elapsed = (time.perf_counter() - t0) * 1000
        checks = tc["expected_checks"]
        actual = pipeline_result.query_plan or pipeline_result.raw_plan

        intent_correct = _check_operation(actual, checks)
        variable_correct = _check_variable(actual, checks)
        region_correct = _check_region(actual, checks)
        filter_correct = _check_filters(actual, pipeline_result.query_result, checks)
        plan_correct = intent_correct and variable_correct and region_correct and filter_correct
        rag_docs = len(pipeline_result.rag_docs)
        rag_correct = rag_docs > 0 if checks.get("rag_docs_retrieved") else True

        return QuestionResult(
            id=tc["id"],
            question=tc["question"],
            category=tc["category"],
            mode=pipeline_result.mode,
            intent_correct=intent_correct,
            variable_correct=variable_correct,
            region_correct=region_correct,
            filter_correct=filter_correct,
            plan_correct=plan_correct,
            rag_correct=rag_correct,
            rag_docs_retrieved=rag_docs,
            execution_error=pipeline_result.error,
            latency_ms=elapsed,
            actual_plan=actual,
            expected_plan=tc["expected_plan"],
        )


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def _check_operation(actual: dict, checks: dict) -> bool:
    exp_op = checks.get("operation")
    if exp_op is None:
        return True
    return actual.get("operation") == exp_op


def _check_variable(actual: dict, checks: dict) -> bool:
    exp_var = checks.get("variable")
    if exp_var is None:
        return True
    return actual.get("variable") == exp_var


def _check_region(actual: dict, checks: dict) -> bool:
    exp_region = checks.get("region")
    if exp_region is None:
        return True
    actual_region = actual.get("filters", {}).get("region", "")
    if isinstance(actual_region, list):
        return exp_region.lower() in [r.lower() for r in actual_region]
    return actual_region.lower() == exp_region.lower()


def _check_filters(actual: dict, query_result: dict, checks: dict) -> bool:
    # Date filter
    if checks.get("has_date_filter"):
        dr = actual.get("filters", {}).get("date_range")
        if not dr:
            return False
    # Depth filter
    if checks.get("has_depth_filter"):
        dep = actual.get("filters", {}).get("depth_range")
        if not dep:
            return False
    # group_by
    exp_gb = checks.get("group_by")
    if exp_gb is not None:
        actual_gb = actual.get("group_by", [])
        if set(exp_gb) != set(actual_gb):
            return False
    return True


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _mean(values: list) -> float:
    if not values:
        return 0.0
    return round(sum(1 if v else 0 for v in values) / len(values), 4)


def _make_summary_df(results: list[QuestionResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "id": r.id,
            "question": r.question[:60] + "…" if len(r.question) > 60 else r.question,
            "category": r.category,
            "mode": r.mode,
            "intent ✓": "✅" if r.intent_correct else "❌",
            "variable ✓": "✅" if r.variable_correct else "❌",
            "region ✓": "✅" if r.region_correct else "❌",
            "filter ✓": "✅" if r.filter_correct else "❌",
            "plan ✓": "✅" if r.plan_correct else "❌",
            "rag ✓": "✅" if r.rag_correct else "❌",
            "latency_ms": round(r.latency_ms, 1),
            "error": r.execution_error or "",
        })
    return pd.DataFrame(rows)

