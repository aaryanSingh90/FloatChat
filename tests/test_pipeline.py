"""
tests/test_pipeline.py
=======================
Integration tests for ai/pipeline.py (without Gemini – rule-based mode)
"""

from __future__ import annotations

import pytest
import pandas as pd
from ai.pipeline import AIPipeline


@pytest.fixture
def pipeline_no_gemini():
    """Pipeline with no Gemini client and no embedder."""
    return AIPipeline(embedder=None, gemini=None)


class TestPipelineRuleBased:
    def test_avg_temperature(self, pipeline_no_gemini, sample_df):
        result = pipeline_no_gemini.process("What is the average temperature?", sample_df)
        assert result.mode == "rule_based"
        assert result.error is None
        assert result.query_result.get("scalar") is not None

    def test_region_filter(self, pipeline_no_gemini, sample_df):
        result = pipeline_no_gemini.process(
            "Show me observations in the North Atlantic", sample_df
        )
        assert result.error is None
        data = result.query_result.get("data", pd.DataFrame())
        # All returned rows should be North Atlantic (or empty df is fine too)
        if not data.empty and "region" in data.columns:
            assert all(data["region"].str.lower() == "north atlantic")

    def test_count_question(self, pipeline_no_gemini, sample_df):
        result = pipeline_no_gemini.process("How many observations are there?", sample_df)
        assert result.error is None
        scalar = result.query_result.get("scalar")
        assert scalar is not None
        assert scalar == 50  # dataset has exactly 50 records

    def test_no_api_key_fallback(self, sample_df):
        """Missing Gemini should not crash the pipeline."""
        from ai.gemini_client import GeminiClient
        import os
        # Ensure no key is set for this test
        old_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            gemini = GeminiClient()
            assert not gemini.is_available
            pipe = AIPipeline(gemini=gemini, embedder=None)
            result = pipe.process("What is the average salinity?", sample_df)
            assert result.mode == "rule_based"
            assert result.error is None
        finally:
            if old_key:
                os.environ["GEMINI_API_KEY"] = old_key

    def test_compare_operation(self, pipeline_no_gemini, sample_df):
        result = pipeline_no_gemini.process("Compare temperature across regions", sample_df)
        assert result.error is None
        data = result.query_result.get("data", pd.DataFrame())
        assert not data.empty

    def test_group_by_region(self, pipeline_no_gemini, sample_df):
        result = pipeline_no_gemini.process("Average temperature by region", sample_df)
        assert result.error is None
        data = result.query_result.get("data", pd.DataFrame())
        assert not data.empty

    def test_deep_filter(self, pipeline_no_gemini, sample_df):
        result = pipeline_no_gemini.process(
            "Show average temperature below 500 m", sample_df
        )
        assert result.error is None

    def test_date_filter(self, pipeline_no_gemini, sample_df):
        result = pipeline_no_gemini.process("Show observations from 2021", sample_df)
        assert result.error is None

    def test_max_salinity(self, pipeline_no_gemini, sample_df):
        result = pipeline_no_gemini.process("What is the maximum salinity?", sample_df)
        assert result.error is None
        scalar = result.query_result.get("scalar")
        assert scalar is not None
        assert scalar == pytest.approx(sample_df["salinity"].max(), rel=1e-3)

    def test_pipeline_plan_echoed(self, pipeline_no_gemini, sample_df):
        result = pipeline_no_gemini.process("Average pressure by region", sample_df)
        assert isinstance(result.query_plan, dict)
        assert "operation" in result.query_plan

