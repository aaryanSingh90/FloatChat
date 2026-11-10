"""
core/query_engine.py
====================
Top-level orchestrator for the query pipeline.

Flow
----
  raw question
       │
       ▼
  AI layer (Gemini or rule-based fallback)  →  raw query plan (dict)
       │
       ▼
  query_validator  →  validated plan
       │
       ▼
  query_optimizer  →  optimised plan
       │
       ▼
  query_executor   →  result dict
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from core.query_validator import validate_query_plan, QueryValidationError
from core.query_optimizer import optimize_query_plan
from core.query_executor import execute_query, QueryExecutionError

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Orchestrates the full query pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        The oceanographic dataset to query.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, raw_plan: dict[str, Any]) -> dict[str, Any]:
        """
        Execute *raw_plan* against the dataset.

        Returns a result dict (see query_executor.execute_query for schema).
        On validation/execution error returns a dict with error key set.
        """
        try:
            validated = validate_query_plan(raw_plan)
        except QueryValidationError as exc:
            logger.warning("Query validation failed: %s | plan=%s", exc, raw_plan)
            return _error_result(str(exc), "validation")

        optimised = optimize_query_plan(validated)
        logger.debug("Optimised plan: %s", optimised)

        try:
            result = execute_query(self._df, optimised)
        except QueryExecutionError as exc:
            logger.error("Query execution failed: %s", exc)
            return _error_result(str(exc), "execution")

        result["plan"] = optimised
        return result

    @property
    def dataset(self) -> pd.DataFrame:
        return self._df


def _error_result(message: str, stage: str) -> dict[str, Any]:
    return {
        "data": pd.DataFrame(),
        "scalar": None,
        "operation": "error",
        "filters_applied": "",
        "row_count": 0,
        "error": f"[{stage}] {message}",
        "plan": {},
    }

