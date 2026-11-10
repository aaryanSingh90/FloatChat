"""
data/generate_dataset.py
========================
Generates a 50-record synthetic Argo-style oceanographic dataset.
Data is clearly labelled as SYNTHETIC DEMONSTRATION DATA and must
not be represented as real Argo scientific observations.

Schema
------
float_id    : unique float identifier (e.g. "WMO_1234567")
latitude    : decimal degrees, -90 to +90
longitude   : decimal degrees, -180 to +180
date        : ISO-8601 date string (YYYY-MM-DD)
temperature : sea-water temperature in °C
salinity    : practical salinity units (PSU)
pressure    : pressure in decibars (dbar)
depth       : approximate depth in metres
region      : named ocean basin / region
qc_flag     : Argo QC flag (1 = good, 2 = probably good, 3 = probably bad, 4 = bad)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Region definitions: (name, lat_range, lon_range)
# ---------------------------------------------------------------------------
REGIONS = [
    ("North Atlantic",    (30,  60),  (-60,  -10)),
    ("South Atlantic",    (-50, -10), (-50,  10)),
    ("North Pacific",     (20,  55),  (140, -120)),   # wraps 180°
    ("South Pacific",     (-60, -10), (-170, -70)),
    ("Indian Ocean",      (-40,  25), (40,   100)),
    ("Southern Ocean",    (-70, -50), (-180,  180)),
    ("Mediterranean Sea", (30,  46),  (-5,   36)),
    ("Arctic Ocean",      (70,  85),  (-180,  180)),
    ("Caribbean Sea",     (10,  25),  (-85,  -60)),
    ("Coral Sea",         (-30, -10), (145,  165)),
]

RECORDS_PER_REGION = 5          # 10 regions × 5 = 50 records exactly


def _lon_in_range(rng, lon_range):
    lo, hi = lon_range
    if lo > hi:                 # wraps across ±180
        lo, hi = hi, lo + 360
    v = rng.uniform(lo, hi)
    if v > 180:
        v -= 360
    return round(float(v), 4)


def _temperature_for_region(rng, region: str, depth: float) -> float:
    """Return a physically plausible temperature (°C) for a region and depth."""
    base_temps = {
        "North Atlantic":    15.0,
        "South Atlantic":    12.0,
        "North Pacific":     14.0,
        "South Pacific":     11.0,
        "Indian Ocean":      22.0,
        "Southern Ocean":     3.0,
        "Mediterranean Sea": 18.0,
        "Arctic Ocean":      -1.5,
        "Caribbean Sea":     26.0,
        "Coral Sea":         24.0,
    }
    surface_t = base_temps.get(region, 10.0)
    deep_drop = min(depth / 1000 * 15, surface_t + 5)
    t = surface_t - deep_drop + rng.normal(0, 0.5)
    return round(float(t), 3)


def _salinity_for_region(rng, region: str) -> float:
    """Return a physically plausible salinity (PSU) for a region."""
    base_sal = {
        "North Atlantic":    35.5,
        "South Atlantic":    35.2,
        "North Pacific":     34.1,
        "South Pacific":     34.5,
        "Indian Ocean":      35.0,
        "Southern Ocean":    34.0,
        "Mediterranean Sea": 38.5,
        "Arctic Ocean":      31.0,
        "Caribbean Sea":     36.2,
        "Coral Sea":         35.4,
    }
    base = base_sal.get(region, 35.0)
    return round(float(base + rng.normal(0, 0.2)), 3)


def generate_dataset(seed: int = 42) -> pd.DataFrame:
    """Generate and return the 50-record synthetic dataset."""
    rng = np.random.default_rng(seed)
    rows = []
    float_counter = 1_000_000

    for region_name, lat_range, lon_range in REGIONS:
        for _ in range(RECORDS_PER_REGION):
            float_counter += rng.integers(100, 500)
            lat = round(float(rng.uniform(*lat_range)), 4)
            lon = _lon_in_range(rng, lon_range)

            # Random date between 2020-01-01 and 2023-12-31
            days_offset = int(rng.integers(0, 365 * 4))
            date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=days_offset)

            pressure = round(float(rng.uniform(10, 2000)), 1)
            # Approx conversion: depth(m) ≈ pressure(dbar) × 0.995
            depth = round(pressure * 0.995, 1)

            temp = _temperature_for_region(rng, region_name, depth)
            sal = _salinity_for_region(rng, region_name)

            # QC flags: mostly good (1), some probably-good (2), rare bad (4)
            qc_flag = int(rng.choice([1, 1, 1, 1, 2, 2, 3, 4], p=[0.5, 0.15, 0.15, 0.1, 0.05, 0.02, 0.02, 0.01]))

            rows.append({
                "float_id":    f"WMO_{float_counter}",
                "latitude":    lat,
                "longitude":   lon,
                "date":        date.strftime("%Y-%m-%d"),
                "temperature": temp,
                "salinity":    sal,
                "pressure":    pressure,
                "depth":       depth,
                "region":      region_name,
                "qc_flag":     qc_flag,
            })

    df = pd.DataFrame(rows)
    df.attrs["description"] = (
        "SYNTHETIC DEMONSTRATION DATA – NOT real Argo observations. "
        "Generated for Float Chat educational / demo purposes only."
    )
    return df


if __name__ == "__main__":
    out_path = Path(__file__).parent / "sample_argo_data.csv"
    df = generate_dataset()
    df.to_csv(out_path, index=False)
    print(f"✅  Wrote {len(df)} synthetic records to {out_path}")
    print(df.head(5).to_string())
