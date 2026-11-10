"""
core/query_executor.py
======================
Executes a validated, optimised query plan against a Pandas DataFrame.

Security note
-------------
This module NEVER calls eval() or exec().  All operations are expressed
as explicit Pandas method calls selected by the allowlisted operation name.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any


class QueryExecutionError(RuntimeError):
    """Raised when execution fails at runtime."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_query(df: pd.DataFrame, plan: dict[str, Any]) -> dict[str, Any]:
    """
    Execute *plan* against *df* and return a result dict containing:
      - data      : pd.DataFrame with result rows
      - scalar    : single numeric result (for single-cell aggregations)
      - operation : echoed operation name
      - filters_applied : human-readable description of applied filters
      - row_count : number of result rows
      - error     : None on success, str on failure
    """
    try:
        working = _apply_filters(df, plan)
        filters_applied = _describe_filters(plan)

        operation = plan.get("operation", "select")
        variable = plan.get("variable")
        group_by = plan.get("group_by")
        sort_by = plan.get("sort_by")
        limit = plan.get("limit", 100)

        if operation == "select":
            result_df = _do_select(working, variable, sort_by, limit)
            return _success(result_df, None, operation, filters_applied)

        if operation == "compare":
            result_df = _do_compare(working, variable)
            return _success(result_df, None, operation, filters_applied)

        # Aggregation operations
        if group_by:
            result_df = _do_group_aggregation(working, operation, variable, group_by, sort_by, limit)
            return _success(result_df, None, operation, filters_applied)
        else:
            scalar, result_df = _do_scalar_aggregation(working, operation, variable)
            return _success(result_df, scalar, operation, filters_applied)

    except QueryExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise QueryExecutionError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

def _apply_filters(df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    filters = plan.get("filters", {})
    mask = pd.Series([True] * len(df), index=df.index)

    # Region
    region = filters.get("region")
    if region:
        if isinstance(region, str):
            mask &= df["region"].str.lower() == region.lower()
        elif isinstance(region, list):
            mask &= df["region"].str.lower().isin([r.lower() for r in region])

    # QC flag
    qc = filters.get("qc_flag")
    if qc is not None:
        if isinstance(qc, int):
            mask &= df["qc_flag"] == qc
        elif isinstance(qc, list):
            mask &= df["qc_flag"].isin(qc)

    # Date range
    date_range = filters.get("date_range", {})
    if date_range:
        dates = pd.to_datetime(df["date"])
        if date_range.get("start"):
            mask &= dates >= pd.to_datetime(date_range["start"])
        if date_range.get("end"):
            mask &= dates <= pd.to_datetime(date_range["end"])

    # Depth range
    depth_range = filters.get("depth_range", {})
    if depth_range:
        if depth_range.get("min") is not None:
            mask &= df["depth"] >= depth_range["min"]
        if depth_range.get("max") is not None:
            mask &= df["depth"] <= depth_range["max"]

    # Pressure range
    pressure_range = filters.get("pressure_range", {})
    if pressure_range:
        if pressure_range.get("min") is not None:
            mask &= df["pressure"] >= pressure_range["min"]
        if pressure_range.get("max") is not None:
            mask &= df["pressure"] <= pressure_range["max"]

    return df[mask].copy()


# ---------------------------------------------------------------------------
# Operation implementations
# ---------------------------------------------------------------------------

def _do_select(df: pd.DataFrame, variable, sort_by, limit: int) -> pd.DataFrame:
    if isinstance(variable, list):
        cols = [c for c in variable if c in df.columns]
        if cols:
            df = df[cols]
    elif isinstance(variable, str) and variable in df.columns:
        # Still return context columns
        ctx_cols = ["float_id", "date", "region", variable]
        df = df[[c for c in ctx_cols if c in df.columns]]
    df = _apply_sort(df, sort_by)
    return df.head(limit)


def _do_compare(df: pd.DataFrame, variable) -> pd.DataFrame:
    col = variable if isinstance(variable, str) else "temperature"
    if col not in df.columns:
        raise QueryExecutionError(f"Column '{col}' not found for comparison.")
    return df.groupby("region")[col].agg(["mean", "min", "max", "count"]).round(3).reset_index()


def _do_group_aggregation(
    df: pd.DataFrame,
    operation: str,
    variable,
    group_by: list[str],
    sort_by,
    limit: int,
) -> pd.DataFrame:
    col = variable if isinstance(variable, str) else _infer_numeric_col(df)
    if col not in df.columns:
        raise QueryExecutionError(f"Column '{col}' not found in dataset.")

    # Support year / month derived columns
    df = _add_time_columns(df, group_by)

    actual_groups = [g for g in group_by if g in df.columns]
    if not actual_groups:
        raise QueryExecutionError(f"None of group_by={group_by} found as columns.")

    agg_fn = _agg_fn(operation)
    if operation == "count":
        result = df.groupby(actual_groups).size().reset_index(name="count")
    else:
        result = df.groupby(actual_groups)[col].agg(agg_fn).reset_index()
        result = result.rename(columns={col: f"{operation}_{col}"})

    result = result.round(4)
    result = _apply_sort(result, sort_by)
    return result.head(limit)


def _do_scalar_aggregation(df: pd.DataFrame, operation: str, variable):
    col = variable if isinstance(variable, str) else _infer_numeric_col(df)
    if col not in df.columns:
        raise QueryExecutionError(f"Column '{col}' not found in dataset.")

    agg_fn = _agg_fn(operation)
    if operation == "count":
        scalar = len(df)
    else:
        scalar = round(float(df[col].agg(agg_fn)), 4)

    result_df = pd.DataFrame({
        "operation": [operation],
        "variable": [col],
        "result": [scalar],
        "n_observations": [len(df)],
    })
    return scalar, result_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agg_fn(operation: str):
    mapping = {
        "avg": "mean",
        "min": "min",
        "max": "max",
        "sum": "sum",
        "median": "median",
        "count": "count",
    }
    return mapping.get(operation, "mean")


def _infer_numeric_col(df: pd.DataFrame) -> str:
    for col in ("temperature", "salinity", "pressure", "depth"):
        if col in df.columns:
            return col
    raise QueryExecutionError("No numeric column available for aggregation.")


def _apply_sort(df: pd.DataFrame, sort_by) -> pd.DataFrame:
    if not sort_by:
        return df
    col = sort_by.get("column")
    order = sort_by.get("order", "asc")
    if col and col in df.columns:
        df = df.sort_values(col, ascending=(order == "asc"))
    return df


def _add_time_columns(df: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    if "year" in group_by or "month" in group_by:
        dates = pd.to_datetime(df["date"])
        if "year" in group_by:
            df = df.copy()
            df["year"] = dates.dt.year
        if "month" in group_by:
            df = df.copy()
            df["month"] = dates.dt.month
    return df


def _describe_filters(plan: dict[str, Any]) -> str:
    filters = plan.get("filters", {})
    parts = []
    if filters.get("region"):
        parts.append(f"region={filters['region']}")
    if filters.get("qc_flag") is not None:
        parts.append(f"qc_flag={filters['qc_flag']}")
    dr = filters.get("date_range", {})
    if dr:
        parts.append(f"date [{dr.get('start','*')} → {dr.get('end','*')}]")
    dep = filters.get("depth_range", {})
    if dep:
        parts.append(f"depth [{dep.get('min','*')} – {dep.get('max','*')} m]")
    return ", ".join(parts) if parts else "none"


def _success(data: pd.DataFrame, scalar, operation: str, filters_applied: str) -> dict[str, Any]:
    return {
        "data": data,
        "scalar": scalar,
        "operation": operation,
        "filters_applied": filters_applied,
        "row_count": len(data),
        "error": None,
    }

