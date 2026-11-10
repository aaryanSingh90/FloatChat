"""
tests/test_intent_parser.py
============================
Unit tests for ai/intent_parser.py (rule-based fallback)
"""

from __future__ import annotations

import pytest
from ai.intent_parser import parse_intent


class TestOperationExtraction:
    def test_avg(self):
        plan = parse_intent("What is the average temperature?")
        assert plan["operation"] == "avg"

    def test_mean(self):
        plan = parse_intent("Show me the mean salinity")
        assert plan["operation"] == "avg"

    def test_maximum(self):
        plan = parse_intent("What is the maximum temperature?")
        assert plan["operation"] == "max"

    def test_minimum(self):
        plan = parse_intent("Find the minimum pressure")
        assert plan["operation"] == "min"

    def test_count(self):
        plan = parse_intent("How many observations are in the dataset?")
        assert plan["operation"] == "count"

    def test_sum(self):
        plan = parse_intent("What is the total depth?")
        assert plan["operation"] == "sum"

    def test_median(self):
        plan = parse_intent("What is the median salinity?")
        assert plan["operation"] == "median"

    def test_compare(self):
        plan = parse_intent("Compare temperature across regions")
        assert plan["operation"] == "compare"

    def test_default_select(self):
        plan = parse_intent("Show me data from the Arctic")
        assert plan["operation"] == "select"


class TestVariableExtraction:
    def test_temperature(self):
        plan = parse_intent("What is the average temperature?")
        assert plan.get("variable") == "temperature"

    def test_salinity(self):
        plan = parse_intent("Show me the salinity distribution")
        assert plan.get("variable") == "salinity"

    def test_pressure(self):
        plan = parse_intent("What is the max pressure?")
        assert plan.get("variable") == "pressure"

    def test_depth(self):
        plan = parse_intent("Count deep observations")
        assert plan.get("variable") == "depth"


class TestRegionExtraction:
    def test_north_atlantic(self):
        plan = parse_intent("Show data from the North Atlantic")
        assert plan["filters"].get("region") == "North Atlantic"

    def test_southern_ocean(self):
        plan = parse_intent("Temperature in the Southern Ocean")
        assert plan["filters"].get("region") == "Southern Ocean"

    def test_mediterranean(self):
        plan = parse_intent("Salinity in the Mediterranean Sea")
        assert plan["filters"].get("region") == "Mediterranean Sea"

    def test_no_region(self):
        plan = parse_intent("What is the average temperature?")
        assert plan["filters"].get("region") is None


class TestFilterExtraction:
    def test_good_qc_flag(self):
        plan = parse_intent("Show me good quality data only")
        assert plan["filters"].get("qc_flag") == [1, 2]

    def test_shallow_depth(self):
        plan = parse_intent("Show temperature shallower than 200 m")
        depth = plan["filters"].get("depth_range")
        assert depth is not None
        assert depth["max"] == 200

    def test_deep_filter(self):
        plan = parse_intent("Find observations below 500 m")
        depth = plan["filters"].get("depth_range")
        assert depth is not None
        assert depth["min"] == 500

    def test_year_filter(self):
        plan = parse_intent("Show all observations in 2021")
        dr = plan["filters"].get("date_range")
        assert dr is not None
        assert "2021" in dr["start"]

    def test_between_years(self):
        plan = parse_intent("Count observations between 2020 and 2022")
        dr = plan["filters"].get("date_range")
        assert dr is not None
        assert "2020" in dr["start"]
        assert "2022" in dr["end"]


class TestGroupByExtraction:
    def test_by_region(self):
        plan = parse_intent("Average temperature by region")
        assert plan.get("group_by") == ["region"]

    def test_by_year(self):
        plan = parse_intent("Count observations per year")
        assert plan.get("group_by") == ["year"]

    def test_by_month(self):
        plan = parse_intent("Average salinity monthly")
        assert plan.get("group_by") == ["month"]


class TestMode:
    def test_always_rule_based(self):
        plan = parse_intent("Any question")
        assert plan["_mode"] == "rule_based"

