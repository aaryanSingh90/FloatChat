"""
tests/conftest.py
=================
Shared pytest fixtures for Float Chat tests.
"""

from __future__ import annotations

import sys
import pathlib

import pytest
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generate_dataset import generate_dataset  # noqa: E402


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """50-record synthetic dataset, shared across all tests."""
    df = generate_dataset()
    # Normalise date to datetime for filter tests
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    """A tiny hand-crafted DataFrame for precise unit tests."""
    return pd.DataFrame([
        {
            "float_id": "WMO_1000001", "latitude": 45.0, "longitude": -30.0,
            "date": pd.Timestamp("2021-06-15"), "temperature": 15.5, "salinity": 35.2,
            "pressure": 100.0, "depth": 99.5, "region": "North Atlantic", "qc_flag": 1,
        },
        {
            "float_id": "WMO_1000002", "latitude": -20.0, "longitude": -40.0,
            "date": pd.Timestamp("2022-01-10"), "temperature": 24.0, "salinity": 36.1,
            "pressure": 10.0, "depth": 9.95, "region": "South Atlantic", "qc_flag": 2,
        },
        {
            "float_id": "WMO_1000003", "latitude": -60.0, "longitude": 0.0,
            "date": pd.Timestamp("2020-11-01"), "temperature": 2.3, "salinity": 33.8,
            "pressure": 1500.0, "depth": 1492.5, "region": "Southern Ocean", "qc_flag": 1,
        },
        {
            "float_id": "WMO_1000004", "latitude": 38.0, "longitude": 20.0,
            "date": pd.Timestamp("2023-07-04"), "temperature": 20.1, "salinity": 38.5,
            "pressure": 50.0, "depth": 49.75, "region": "Mediterranean Sea", "qc_flag": 4,
        },
    ])

