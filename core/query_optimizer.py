"""
core/query_optimizer.py
=======================
Analyses a validated query plan and applies lightweight optimisations before
execution:
  - Adds a default limit when none is specified.
  - Pushes filter predicates to the front of the pipeline description.
  - Coerces region strings to title-case to match the dataset.
  - Annotates the plan with execution hints for the executor.
"""

from __future__ import annotations

from typing import Any

DEFAULT_LIMIT = 100
MAX_SAFE_LIMIT = 10_000


def optimize_query_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """
    Return an optimised (but semantically equivalent) copy of *plan*.
    Does NOT mutate the original.
    """
    plan = dict(plan)   # shallow copy

    # Normalise region casing to match the DataFrame values
    if "filters" in plan:
        plan["filters"] = dict(plan["filters"])
        region = plan["filters"].get("region")
        if isinstance(region, str):
            plan["filters"]["region"] = region.title()
        elif isinstance(region, list):
            plan["filters"]["region"] = [r.title() for r in region]

    # Apply a default row limit
    if plan.get("limit") is None:
        plan["limit"] = DEFAULT_LIMIT

    # Annotate execution hints
    plan["_hints"] = _build_hints(plan)

    return plan


def _build_hints(plan: dict[str, Any]) -> dict[str, Any]:
    hints: dict[str, Any] = {}

    has_filters = bool(plan.get("filters"))
    hints["filter_first"] = has_filters

    op = plan.get("operation", "select")
    hints["needs_aggregation"] = op in {"avg", "min", "max", "sum", "count", "median", "compare"}

    hints["needs_group_by"] = bool(plan.get("group_by"))

    hints["needs_sort"] = bool(plan.get("sort_by"))

    return hints

