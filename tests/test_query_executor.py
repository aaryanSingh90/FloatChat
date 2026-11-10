"""
tests/test_query_executor.py
=============================
Unit tests for core/query_executor.py
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from core.query_executor import execute_query, QueryExecutionError


class TestAggregations:
    def test_avg_temperature(self, small_df):
        plan = {
            "operation": "avg", "variable": "temperature",
            "filters": {}, "_hints": {"needs_aggregation": True}, "limit": 100,
        }
        result = execute_query(small_df, plan)
        assert result["error"] is None
        expected = small_df["temperature"].mean()
        assert abs(result["scalar"] - expected) < 0.0001

    def test_max_salinity(self, small_df):
        plan = {
            "operation": "max", "variable": "salinity",
            "filters": {}, "_hints": {}, "limit": 100,
        }
        result = execute_query(small_df, plan)
        assert result["scalar"] == pytest.approx(small_df["salinity"].max(), rel=1e-4)

    def test_min_temperature(self, small_df):
        plan = {
            "operation": "min", "variable": "temperature",
            "filters": {}, "_hints": {}, "limit": 100,
        }
        result = execute_query(small_df, plan)
        assert result["scalar"] == pytest.approx(small_df["temperature"].min(), rel=1e-4)

    def test_count(self, small_df):
        plan = {
            "operation": "count", "variable": "float_id",
            "filters": {}, "_hints": {}, "limit": 100,
        }
        result = execute_query(small_df, plan)
        assert result["scalar"] == 4

    def test_sum_depth(self, small_df):
        plan = {
            "operation": "sum", "variable": "depth",
            "filters": {}, "_hints": {}, "limit": 100,
        }
        result = execute_query(small_df, plan)
        assert result["scalar"] == pytest.approx(small_df["depth"].sum(), rel=1e-4)

    def test_median_pressure(self, small_df):
        plan = {
            "operation": "median", "variable": "pressure",
            "filters": {}, "_hints": {}, "limit": 100,
        }
        result = execute_query(small_df, plan)
        assert result["scalar"] == pytest.approx(small_df["pressure"].median(), rel=1e-4)


class TestFiltering:
    def test_region_filter(self, small_df):
        plan = {
            "operation": "avg", "variable": "temperature",
            "filters": {"region": "North Atlantic"}, "_hints": {}, "limit": 100,
        }
        result = execute_query(small_df, plan)
        assert result["error"] is None
        # Only WMO_1000001 is North Atlantic, temp = 15.5
        assert result["scalar"] == pytest.approx(15.5, rel=1e-4)

    def test_qc_flag_filter(self, small_df):
        plan = {
            "operation": "count", "variable": "float_id",
            "filters": {"qc_flag": [1]}, "_hints": {}, "limit": 100,
        }
        result = execute_query(small_df, plan)
        expected = len(small_df[small_df["qc_flag"] == 1])
        assert result["scalar"] == expected

    def test_depth_range_filter(self, small_df):
        plan = {
            "operation": "count", "variable": "float_id",
            "filters": {"depth_range": {"min": 0, "max": 200}}, "_hints": {}, "limit": 100,
        }
        result = execute_query(small_df, plan)
        expected = len(small_df[small_df["depth"] <= 200])
        assert result["scalar"] == expected

    def test_date_range_filter(self, small_df):
        plan = {
            "operation": "count", "variable": "float_id",
            "filters": {"date_range": {"start": "2021-01-01", "end": "2021-12-31"}},
            "_hints": {}, "limit": 100,
        }
        result = execute_query(small_df, plan)
        # Only WMO_1000001 is 2021-06-15
        assert result["scalar"] == 1

    def test_no_matching_rows(self, small_df):
        plan = {
            "operation": "count", "variable": "float_id",
            "filters": {"region": "Arctic Ocean"}, "_hints": {}, "limit": 100,
        }
        result = execute_query(small_df, plan)
        assert result["scalar"] == 0


class TestGroupBy:
    def test_group_by_region(self, sample_df):
        # Need datetime date for the executor date filter
        df = sample_df.copy()
        df["date"] = df["date"].astype(str)
        plan = {
            "operation": "avg", "variable": "temperature",
            "filters": {}, "group_by": ["region"],
            "_hints": {"needs_group_by": True}, "limit": 100,
        }
        result = execute_query(sample_df, plan)
        assert result["error"] is None
        assert not result["data"].empty
        assert "region" in result["data"].columns

    def test_count_by_region(self, sample_df):
        plan = {
            "operation": "count", "variable": "float_id",
            "filters": {}, "group_by": ["region"],
            "_hints": {"needs_group_by": True}, "limit": 100,
        }
        result = execute_query(sample_df, plan)
        assert result["error"] is None
        assert "count" in result["data"].columns
        # Should have 10 regions
        assert len(result["data"]) == 10


class TestSelect:
    def test_select_returns_rows(self, small_df):
        plan = {
            "operation": "select", "variable": "temperature",
            "filters": {}, "_hints": {}, "limit": 10,
        }
        result = execute_query(small_df, plan)
        assert result["error"] is None
        assert not result["data"].empty

    def test_limit_respected(self, sample_df):
        plan = {
            "operation": "select", "variable": None,
            "filters": {}, "_hints": {}, "limit": 5,
        }
        result = execute_query(sample_df, plan)
        assert result["row_count"] == 5


class TestCompare:
    def test_compare_returns_all_regions(self, sample_df):
        plan = {
            "operation": "compare", "variable": "temperature",
            "filters": {}, "_hints": {}, "limit": 100,
        }
        result = execute_query(sample_df, plan)
        assert result["error"] is None
        assert len(result["data"]) == 10   # 10 regions
        assert "mean" in result["data"].columns

