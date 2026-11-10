"""
core/query_validator.py
=======================
Validates a query plan produced by the AI layer against a strict allowlist.
No arbitrary code is ever executed — only plans matching the schema reach
the executor.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------
ALLOWED_OPERATIONS = {"avg", "min", "max", "sum", "count", "median", "select", "compare"}

ALLOWED_COLUMNS = {
    "float_id", "latitude", "longitude", "date",
    "temperature", "salinity", "pressure", "depth",
    "region", "qc_flag",
}

NUMERIC_COLUMNS = {"latitude", "longitude", "temperature", "salinity", "pressure", "depth"}

ALLOWED_GROUP_BY = {"region", "qc_flag", "year", "month", "date"}

ALLOWED_SORT_COLUMNS = ALLOWED_COLUMNS | {"year", "month"}

ALLOWED_SORT_ORDERS = {"asc", "desc"}

ALLOWED_REGIONS = {
    "north atlantic", "south atlantic", "north pacific", "south pacific",
    "indian ocean", "southern ocean", "mediterranean sea", "arctic ocean",
    "caribbean sea", "coral sea",
}

ALLOWED_QC_FLAGS = {0, 1, 2, 3, 4, 5, 8, 9}


class QueryValidationError(ValueError):
    """Raised when a query plan fails validation."""


def validate_query_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """
    Validate *plan* against the allowlist schema.

    Returns the (possibly normalised) plan if valid.
    Raises QueryValidationError with a descriptive message otherwise.
    """
    if not isinstance(plan, dict):
        raise QueryValidationError("Query plan must be a dict.")

    _validate_operation(plan)
    _validate_variable(plan)
    _validate_group_by(plan)
    _validate_sort(plan)
    _validate_filters(plan)
    _validate_limit(plan)

    return plan


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_operation(plan: dict) -> None:
    op = plan.get("operation", "").lower()
    if op not in ALLOWED_OPERATIONS:
        raise QueryValidationError(
            f"Invalid operation '{op}'. Allowed: {sorted(ALLOWED_OPERATIONS)}"
        )
    plan["operation"] = op


def _validate_variable(plan: dict) -> None:
    variable = plan.get("variable")
    if variable is None:
        return
    if isinstance(variable, str):
        if variable.lower() not in ALLOWED_COLUMNS:
            raise QueryValidationError(
                f"Invalid variable '{variable}'. Allowed: {sorted(ALLOWED_COLUMNS)}"
            )
        plan["variable"] = variable.lower()
    elif isinstance(variable, list):
        for v in variable:
            if v.lower() not in ALLOWED_COLUMNS:
                raise QueryValidationError(
                    f"Invalid variable '{v}'. Allowed: {sorted(ALLOWED_COLUMNS)}"
                )
        plan["variable"] = [v.lower() for v in variable]


def _validate_group_by(plan: dict) -> None:
    gb = plan.get("group_by")
    if gb is None:
        return
    if isinstance(gb, str):
        gb = [gb]
    for g in gb:
        if g.lower() not in ALLOWED_GROUP_BY:
            raise QueryValidationError(
                f"Invalid group_by value '{g}'. Allowed: {sorted(ALLOWED_GROUP_BY)}"
            )
    plan["group_by"] = [g.lower() for g in gb]


def _validate_sort(plan: dict) -> None:
    sort = plan.get("sort_by")
    if sort is None:
        return
    if sort.get("column", "").lower() not in ALLOWED_SORT_COLUMNS:
        raise QueryValidationError(
            f"Invalid sort column '{sort.get('column')}'. Allowed: {sorted(ALLOWED_SORT_COLUMNS)}"
        )
    order = sort.get("order", "asc").lower()
    if order not in ALLOWED_SORT_ORDERS:
        raise QueryValidationError(f"Invalid sort order '{order}'. Use 'asc' or 'desc'.")
    plan["sort_by"]["column"] = sort["column"].lower()
    plan["sort_by"]["order"] = order


def _validate_filters(plan: dict) -> None:
    filters = plan.get("filters", {})
    if not isinstance(filters, dict):
        raise QueryValidationError("'filters' must be a dict.")

    # Region
    region = filters.get("region")
    if region is not None:
        if isinstance(region, str):
            if region.lower() not in ALLOWED_REGIONS:
                raise QueryValidationError(
                    f"Unknown region '{region}'. Allowed: {sorted(ALLOWED_REGIONS)}"
                )
            filters["region"] = region.lower()
        elif isinstance(region, list):
            for r in region:
                if r.lower() not in ALLOWED_REGIONS:
                    raise QueryValidationError(
                        f"Unknown region '{r}'. Allowed: {sorted(ALLOWED_REGIONS)}"
                    )
            filters["region"] = [r.lower() for r in region]

    # Depth range
    depth_range = filters.get("depth_range")
    if depth_range is not None:
        _validate_numeric_range(depth_range, "depth_range", 0, 11000)

    # Pressure range
    pressure_range = filters.get("pressure_range")
    if pressure_range is not None:
        _validate_numeric_range(pressure_range, "pressure_range", 0, 11000)

    # Date range
    date_range = filters.get("date_range")
    if date_range is not None:
        if not isinstance(date_range, dict):
            raise QueryValidationError("'date_range' must be a dict with 'start' and/or 'end'.")

    # QC flag
    qc = filters.get("qc_flag")
    if qc is not None:
        if isinstance(qc, int):
            if qc not in ALLOWED_QC_FLAGS:
                raise QueryValidationError(f"Invalid qc_flag {qc}. Allowed: {sorted(ALLOWED_QC_FLAGS)}")
        elif isinstance(qc, list):
            for q in qc:
                if q not in ALLOWED_QC_FLAGS:
                    raise QueryValidationError(f"Invalid qc_flag {q}. Allowed: {sorted(ALLOWED_QC_FLAGS)}")

    plan["filters"] = filters


def _validate_numeric_range(rng: dict, name: str, lo: float, hi: float) -> None:
    if not isinstance(rng, dict):
        raise QueryValidationError(f"'{name}' must be a dict with 'min' and/or 'max'.")
    mn = rng.get("min")
    mx = rng.get("max")
    if mn is not None and not isinstance(mn, (int, float)):
        raise QueryValidationError(f"'{name}.min' must be numeric.")
    if mx is not None and not isinstance(mx, (int, float)):
        raise QueryValidationError(f"'{name}.max' must be numeric.")
    if mn is not None and not (lo <= mn <= hi):
        raise QueryValidationError(f"'{name}.min' = {mn} out of range [{lo}, {hi}].")
    if mx is not None and not (lo <= mx <= hi):
        raise QueryValidationError(f"'{name}.max' = {mx} out of range [{lo}, {hi}].")
    if mn is not None and mx is not None and mn > mx:
        raise QueryValidationError(f"'{name}.min' > '{name}.max'.")


def _validate_limit(plan: dict) -> None:
    limit = plan.get("limit")
    if limit is None:
        return
    if not isinstance(limit, int) or limit < 1 or limit > 10_000:
        raise QueryValidationError("'limit' must be an integer between 1 and 10000.")

