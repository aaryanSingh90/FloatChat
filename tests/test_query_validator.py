"""
tests/test_query_validator.py
==============================
Unit tests for core/query_validator.py
"""

from __future__ import annotations

import pytest
from core.query_validator import validate_query_plan, QueryValidationError


# ---------------------------------------------------------------------------
# Valid plans
# ---------------------------------------------------------------------------

class TestValidPlans:
    def test_simple_avg(self):
        plan = {"operation": "avg", "variable": "temperature", "filters": {}}
        result = validate_query_plan(plan)
        assert result["operation"] == "avg"
        assert result["variable"] == "temperature"

    def test_max_with_region_filter(self):
        plan = {
            "operation": "max",
            "variable": "salinity",
            "filters": {"region": "north atlantic"},
        }
        result = validate_query_plan(plan)
        assert result["operation"] == "max"

    def test_count_no_variable(self):
        plan = {"operation": "count", "filters": {}}
        result = validate_query_plan(plan)
        assert result["operation"] == "count"

    def test_select_operation(self):
        plan = {"operation": "select", "variable": "temperature", "filters": {}}
        result = validate_query_plan(plan)
        assert result["operation"] == "select"

    def test_group_by_region(self):
        plan = {
            "operation": "avg",
            "variable": "temperature",
            "filters": {},
            "group_by": ["region"],
        }
        result = validate_query_plan(plan)
        assert result["group_by"] == ["region"]

    def test_sort_descending(self):
        plan = {
            "operation": "select",
            "variable": "temperature",
            "filters": {},
            "sort_by": {"column": "temperature", "order": "desc"},
        }
        result = validate_query_plan(plan)
        assert result["sort_by"]["order"] == "desc"

    def test_valid_limit(self):
        plan = {"operation": "select", "filters": {}, "limit": 50}
        result = validate_query_plan(plan)
        assert result["limit"] == 50

    def test_depth_range_filter(self):
        plan = {
            "operation": "avg",
            "variable": "temperature",
            "filters": {"depth_range": {"min": 100, "max": 500}},
        }
        result = validate_query_plan(plan)
        assert result["filters"]["depth_range"]["min"] == 100

    def test_date_range_filter(self):
        plan = {
            "operation": "count",
            "filters": {"date_range": {"start": "2021-01-01", "end": "2021-12-31"}},
        }
        result = validate_query_plan(plan)
        assert result["filters"]["date_range"]["start"] == "2021-01-01"

    def test_qc_flag_list(self):
        plan = {
            "operation": "avg",
            "variable": "temperature",
            "filters": {"qc_flag": [1, 2]},
        }
        result = validate_query_plan(plan)
        assert 1 in result["filters"]["qc_flag"]

    def test_compare_operation(self):
        plan = {"operation": "compare", "variable": "temperature", "filters": {}}
        result = validate_query_plan(plan)
        assert result["operation"] == "compare"

    def test_median_operation(self):
        plan = {"operation": "median", "variable": "pressure", "filters": {}}
        result = validate_query_plan(plan)
        assert result["operation"] == "median"


# ---------------------------------------------------------------------------
# Invalid plans — should raise QueryValidationError
# ---------------------------------------------------------------------------

class TestInvalidPlans:
    def test_invalid_operation(self):
        with pytest.raises(QueryValidationError, match="Invalid operation"):
            validate_query_plan({"operation": "DELETE", "filters": {}})

    def test_invalid_variable(self):
        with pytest.raises(QueryValidationError, match="Invalid variable"):
            validate_query_plan({"operation": "avg", "variable": "password", "filters": {}})

    def test_invalid_region(self):
        with pytest.raises(QueryValidationError, match="Unknown region"):
            validate_query_plan({
                "operation": "avg",
                "variable": "temperature",
                "filters": {"region": "Narnia"},
            })

    def test_invalid_group_by(self):
        with pytest.raises(QueryValidationError, match="Invalid group_by"):
            validate_query_plan({
                "operation": "avg",
                "variable": "temperature",
                "filters": {},
                "group_by": ["hack"],
            })

    def test_invalid_sort_column(self):
        with pytest.raises(QueryValidationError, match="Invalid sort column"):
            validate_query_plan({
                "operation": "select",
                "filters": {},
                "sort_by": {"column": "exec()", "order": "asc"},
            })

    def test_invalid_sort_order(self):
        with pytest.raises(QueryValidationError, match="Invalid sort order"):
            validate_query_plan({
                "operation": "select",
                "filters": {},
                "sort_by": {"column": "temperature", "order": "RANDOM"},
            })

    def test_depth_range_inverted(self):
        with pytest.raises(QueryValidationError, match="min.*max"):
            validate_query_plan({
                "operation": "avg",
                "variable": "temperature",
                "filters": {"depth_range": {"min": 500, "max": 100}},
            })

    def test_limit_too_large(self):
        with pytest.raises(QueryValidationError, match="limit"):
            validate_query_plan({"operation": "select", "filters": {}, "limit": 99999})

    def test_limit_zero(self):
        with pytest.raises(QueryValidationError, match="limit"):
            validate_query_plan({"operation": "select", "filters": {}, "limit": 0})

    def test_not_a_dict(self):
        with pytest.raises(QueryValidationError):
            validate_query_plan("DROP TABLE users")  # type: ignore[arg-type]

    def test_invalid_qc_flag(self):
        with pytest.raises(QueryValidationError, match="qc_flag"):
            validate_query_plan({
                "operation": "count",
                "filters": {"qc_flag": 99},
            })

