"""
ai/intent_parser.py
====================
Rule-based fallback intent parser.

Parses a natural-language question into a structured query plan (dict)
without calling any external API.  Used when the Gemini API key is absent
or the API call fails.

Supported patterns
------------------
- Aggregation keywords: average, mean, min, max, minimum, maximum,
  count, sum, total, median
- Variables: temperature, salinity, pressure, depth, latitude, longitude
- Regions: north atlantic, south atlantic, north pacific, south pacific,
           indian ocean, southern ocean, mediterranean sea, arctic ocean,
           caribbean sea, coral sea
- Depth modifiers: shallow / deep / surface / below X m / above X m / 0–500 m
- Date modifiers: in 2021, between 2020 and 2022, after January 2022
- QC filters: good data only / qc=1
- Group by: by region, by year, by month
- Sort: highest, lowest, ordered by
"""

from __future__ import annotations

import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------

OPERATION_KEYWORDS: dict[str, str] = {
    r"\baverage\b|\bmean\b|\bavg\b":          "avg",
    r"\bminimum\b|\bmin\b|\blowest\b|\bcoldest\b|\bshallower\b|\bleast\b": "min",
    r"\bmaximum\b|\bmax\b|\bhighest\b|\bwarmest\b|\bdeepest\b|\bmost\b":   "max",
    r"\bcount\b|\bhow many\b|\bnumber of\b":  "count",
    r"\bsum\b|\btotal\b":                      "sum",
    r"\bmedian\b":                             "median",
    r"\bcompare\b|\bcomparison\b|\bversus\b|\bvs\.?\b": "compare",
}

VARIABLE_KEYWORDS: dict[str, str] = {
    r"\btemperature\b|\btemp\b|\bthermal\b":  "temperature",
    r"\bsalinity\b|\bsalt\b|\bsaline\b":      "salinity",
    r"\bpressure\b|\bdbar\b":                  "pressure",
    r"\bdepth\b|\bdeep\b|\bshallow\b|\bmeters?\b|\bmetres?\b": "depth",
    r"\blatitude\b|\blat\b":                   "latitude",
    r"\blongitude\b|\blon\b|\blong\b":         "longitude",
    r"\bfloat\b|\bfloats?\b|\bwmo\b":          "float_id",
    r"\bqc.flag\b|\bquality\b":                "qc_flag",
}

REGION_KEYWORDS: dict[str, str] = {
    r"north\s+atlantic":    "North Atlantic",
    r"south\s+atlantic":    "South Atlantic",
    r"north\s+pacific":     "North Pacific",
    r"south\s+pacific":     "South Pacific",
    r"indian\s+ocean":      "Indian Ocean",
    r"southern\s+ocean":    "Southern Ocean",
    r"mediterranean":       "Mediterranean Sea",
    r"arctic\s+ocean":      "Arctic Ocean",
    r"caribbean":           "Caribbean Sea",
    r"coral\s+sea":         "Coral Sea",
}

GROUP_BY_KEYWORDS: dict[str, str] = {
    r"\bby\s+region\b|\bper\s+region\b|\beach\s+region\b|\bregional\b": "region",
    r"\bby\s+year\b|\bper\s+year\b|\bannual\b|\byearly\b":               "year",
    r"\bby\s+month\b|\bper\s+month\b|\bmonthly\b":                      "month",
    r"\bby\s+qc\b|\bby\s+flag\b":                                        "qc_flag",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_intent(question: str) -> dict[str, Any]:
    """
    Parse *question* using rule-based heuristics.

    Returns a query-plan dict compatible with the validator schema.
    The plan always includes a '_mode' key set to 'rule_based'.
    """
    q = question.lower()

    plan: dict[str, Any] = {
        "_mode": "rule_based",
        "operation": _extract_operation(q),
        "variable": _extract_variable(q),
        "filters": _extract_filters(q),
        "group_by": _extract_group_by(q),
        "sort_by": _extract_sort(q),
        "limit": _extract_limit(q),
    }

    # Remove None values to keep the plan lean
    plan = {k: v for k, v in plan.items() if v is not None or k in ("_mode", "filters")}
    if not plan.get("filters"):
        plan["filters"] = {}

    logger.debug("Rule-based plan: %s", plan)
    return plan


# ---------------------------------------------------------------------------
# Private extractors
# ---------------------------------------------------------------------------

def _extract_operation(q: str) -> str:
    for pattern, op in OPERATION_KEYWORDS.items():
        if re.search(pattern, q):
            return op
    return "select"


def _extract_variable(q: str) -> Optional[str]:
    for pattern, var in VARIABLE_KEYWORDS.items():
        if re.search(pattern, q):
            return var
    return None


def _extract_filters(q: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}

    # Region
    for pattern, region in REGION_KEYWORDS.items():
        if re.search(pattern, q, re.IGNORECASE):
            filters["region"] = region
            break

    # QC flag
    if re.search(r"\bgood\s+(?:quality\s+)?data\b|\bqc\s*[=:]\s*1\b|\bflag\s*1\b|\bqc_flag\s*1\b", q):
        filters["qc_flag"] = [1, 2]
    elif re.search(r"\bqc\s*[=:]\s*(\d)\b", q):
        m = re.search(r"\bqc\s*[=:]\s*(\d)\b", q)
        filters["qc_flag"] = int(m.group(1))  # type: ignore[union-attr]

    # Depth range
    depth_range = _parse_depth_range(q)
    if depth_range:
        filters["depth_range"] = depth_range

    # Date range
    date_range = _parse_date_range(q)
    if date_range:
        filters["date_range"] = date_range

    return filters


def _parse_depth_range(q: str) -> Optional[dict]:
    # Explicit "between X and Y m"
    m = re.search(r"between\s+(\d+)\s+and\s+(\d+)\s*(m|meters?|metres?)?", q)
    if m and re.search(r"depth|shallow|deep|meter|metre|m\b", q):
        return {"min": float(m.group(1)), "max": float(m.group(2))}

    # "below X m" / "deeper than X"
    m = re.search(r"(?:below|deeper\s+than|greater\s+than)\s+(\d+)\s*(?:m|meters?|metres?|dbar)?", q)
    if m:
        return {"min": float(m.group(1)), "max": None}

    # "above X m" / "shallower than X"
    m = re.search(r"(?:above|shallower\s+than|less\s+than)\s+(\d+)\s*(?:m|meters?|metres?|dbar)?", q)
    if m:
        return {"min": None, "max": float(m.group(1))}

    # "surface" → 0–200 m
    if re.search(r"\bsurface\b", q):
        return {"min": 0, "max": 200}

    # "shallow" → 0–500 m
    if re.search(r"\bshallow\b", q):
        return {"min": 0, "max": 500}

    # "deep" → >500 m
    if re.search(r"\bdeep\b", q):
        return {"min": 500, "max": None}

    return None


def _parse_date_range(q: str) -> Optional[dict]:
    # "in YYYY"
    m = re.search(r"\bin\s+(20\d{2})\b", q)
    if m:
        yr = m.group(1)
        return {"start": f"{yr}-01-01", "end": f"{yr}-12-31"}

    # "between YYYY and YYYY"
    m = re.search(r"between\s+(20\d{2})\s+and\s+(20\d{2})", q)
    if m:
        return {"start": f"{m.group(1)}-01-01", "end": f"{m.group(2)}-12-31"}

    # "after YYYY" / "since YYYY"
    m = re.search(r"(?:after|since)\s+(20\d{2})", q)
    if m:
        return {"start": f"{m.group(1)}-01-01", "end": None}

    # "before YYYY"
    m = re.search(r"before\s+(20\d{2})", q)
    if m:
        return {"start": None, "end": f"{m.group(1)}-12-31"}

    return None


def _extract_group_by(q: str) -> Optional[list[str]]:
    for pattern, group in GROUP_BY_KEYWORDS.items():
        if re.search(pattern, q):
            return [group]
    return None


def _extract_sort(q: str) -> Optional[dict]:
    if re.search(r"\bhighest\b|\bmost\b|\blargest\b|\bwarmest\b|\bsaltest\b", q):
        return {"column": "temperature", "order": "desc"}
    if re.search(r"\blowest\b|\bleast\b|\bcoldest\b|\bsmallest\b", q):
        return {"column": "temperature", "order": "asc"}
    if re.search(r"\bsort(?:ed)?\s+by\s+(\w+)\b", q):
        m = re.search(r"\bsort(?:ed)?\s+by\s+(\w+)\b", q)
        col = m.group(1) if m else "temperature"  # type: ignore[union-attr]
        return {"column": col, "order": "asc"}
    return None


def _extract_limit(q: str) -> Optional[int]:
    m = re.search(r"\btop\s+(\d+)\b|\bfirst\s+(\d+)\b|\blimit\s+(\d+)\b", q)
    if m:
        val = m.group(1) or m.group(2) or m.group(3)
        return int(val)
    return None

