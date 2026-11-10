"""
tests/test_dataset.py
======================
Tests for the synthetic dataset generation.
"""

from __future__ import annotations

import pandas as pd
import pytest
from data.generate_dataset import generate_dataset

REQUIRED_COLUMNS = {
    "float_id", "latitude", "longitude", "date",
    "temperature", "salinity", "pressure", "depth",
    "region", "qc_flag",
}
VALID_QC_FLAGS = {0, 1, 2, 3, 4, 5, 8, 9}
KNOWN_REGIONS = {
    "North Atlantic", "South Atlantic", "North Pacific", "South Pacific",
    "Indian Ocean", "Southern Ocean", "Mediterranean Sea", "Arctic Ocean",
    "Caribbean Sea", "Coral Sea",
}


class TestDatasetSchema:
    def test_exactly_50_records(self):
        df = generate_dataset()
        assert len(df) == 50

    def test_required_columns_present(self):
        df = generate_dataset()
        assert REQUIRED_COLUMNS.issubset(df.columns)

    def test_no_null_values(self):
        df = generate_dataset()
        assert df.isnull().sum().sum() == 0

    def test_latitude_range(self):
        df = generate_dataset()
        assert df["latitude"].between(-90, 90).all()

    def test_longitude_range(self):
        df = generate_dataset()
        assert df["longitude"].between(-180, 180).all()

    def test_valid_qc_flags(self):
        df = generate_dataset()
        assert set(df["qc_flag"].unique()).issubset(VALID_QC_FLAGS)

    def test_all_regions_represented(self):
        df = generate_dataset()
        assert set(df["region"].unique()) == KNOWN_REGIONS

    def test_exactly_5_per_region(self):
        df = generate_dataset()
        counts = df["region"].value_counts()
        assert (counts == 5).all()

    def test_pressure_non_negative(self):
        df = generate_dataset()
        assert (df["pressure"] >= 0).all()

    def test_depth_proportional_to_pressure(self):
        df = generate_dataset()
        # depth ≈ pressure × 0.995
        ratio = df["depth"] / df["pressure"]
        assert ratio.between(0.99, 1.0).all()

    def test_date_format_valid(self):
        df = generate_dataset()
        # Must parse without errors
        pd.to_datetime(df["date"])

    def test_reproducibility(self):
        df1 = generate_dataset()
        df2 = generate_dataset()
        assert df1["temperature"].tolist() == df2["temperature"].tolist()

