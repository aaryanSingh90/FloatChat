"""
evaluation/test_questions.py
=============================
30+ natural-language test questions with ground-truth query plans and
expected answers (where deterministic from the synthetic dataset).

Each test case is a dict with:
  id              : unique identifier
  question        : natural-language question
  expected_plan   : ground-truth query plan dict
  expected_checks : dict of key → value or callable(result) → bool
  category        : one of intent | variable | region | filter | aggregation | rag
"""

from __future__ import annotations

TEST_QUESTIONS: list[dict] = [
    # ------------------------------------------------------------------ AVG
    {
        "id": "q01",
        "question": "What is the average temperature across all observations?",
        "expected_plan": {"operation": "avg", "variable": "temperature", "filters": {}},
        "expected_checks": {"operation": "avg", "variable": "temperature"},
        "category": "aggregation",
    },
    {
        "id": "q02",
        "question": "What is the mean salinity in the North Atlantic?",
        "expected_plan": {
            "operation": "avg",
            "variable": "salinity",
            "filters": {"region": "North Atlantic"},
        },
        "expected_checks": {"operation": "avg", "variable": "salinity", "region": "North Atlantic"},
        "category": "region",
    },
    {
        "id": "q03",
        "question": "Calculate average depth in the Southern Ocean",
        "expected_plan": {
            "operation": "avg",
            "variable": "depth",
            "filters": {"region": "Southern Ocean"},
        },
        "expected_checks": {"operation": "avg", "variable": "depth", "region": "Southern Ocean"},
        "category": "aggregation",
    },
    # ------------------------------------------------------------------ MAX
    {
        "id": "q04",
        "question": "What is the maximum temperature recorded?",
        "expected_plan": {"operation": "max", "variable": "temperature", "filters": {}},
        "expected_checks": {"operation": "max", "variable": "temperature"},
        "category": "aggregation",
    },
    {
        "id": "q05",
        "question": "Find the warmest observation in the Caribbean Sea",
        "expected_plan": {
            "operation": "max",
            "variable": "temperature",
            "filters": {"region": "Caribbean Sea"},
        },
        "expected_checks": {"operation": "max", "variable": "temperature", "region": "Caribbean Sea"},
        "category": "region",
    },
    {
        "id": "q06",
        "question": "What is the maximum salinity in the Mediterranean Sea?",
        "expected_plan": {
            "operation": "max",
            "variable": "salinity",
            "filters": {"region": "Mediterranean Sea"},
        },
        "expected_checks": {"operation": "max", "variable": "salinity", "region": "Mediterranean Sea"},
        "category": "region",
    },
    # ------------------------------------------------------------------ MIN
    {
        "id": "q07",
        "question": "What is the minimum temperature observed?",
        "expected_plan": {"operation": "min", "variable": "temperature", "filters": {}},
        "expected_checks": {"operation": "min", "variable": "temperature"},
        "category": "aggregation",
    },
    {
        "id": "q08",
        "question": "Find the coldest temperature in the Arctic Ocean",
        "expected_plan": {
            "operation": "min",
            "variable": "temperature",
            "filters": {"region": "Arctic Ocean"},
        },
        "expected_checks": {"operation": "min", "variable": "temperature", "region": "Arctic Ocean"},
        "category": "region",
    },
    {
        "id": "q09",
        "question": "What is the lowest salinity recorded?",
        "expected_plan": {"operation": "min", "variable": "salinity", "filters": {}},
        "expected_checks": {"operation": "min", "variable": "salinity"},
        "category": "aggregation",
    },
    # ---------------------------------------------------------------- COUNT
    {
        "id": "q10",
        "question": "How many observations are in the dataset?",
        "expected_plan": {"operation": "count", "variable": "float_id", "filters": {}},
        "expected_checks": {"operation": "count"},
        "category": "intent",
    },
    {
        "id": "q11",
        "question": "Count observations in the Indian Ocean",
        "expected_plan": {
            "operation": "count",
            "variable": "float_id",
            "filters": {"region": "Indian Ocean"},
        },
        "expected_checks": {"operation": "count", "region": "Indian Ocean"},
        "category": "filter",
    },
    {
        "id": "q12",
        "question": "How many good quality observations are there?",
        "expected_plan": {
            "operation": "count",
            "variable": "float_id",
            "filters": {"qc_flag": [1, 2]},
        },
        "expected_checks": {"operation": "count"},
        "category": "filter",
    },
    # --------------------------------------------------------------- MEDIAN
    {
        "id": "q13",
        "question": "What is the median pressure in the dataset?",
        "expected_plan": {"operation": "median", "variable": "pressure", "filters": {}},
        "expected_checks": {"operation": "median", "variable": "pressure"},
        "category": "aggregation",
    },
    # ------------------------------------------------------------------ SUM
    {
        "id": "q14",
        "question": "What is the total depth across all observations?",
        "expected_plan": {"operation": "sum", "variable": "depth", "filters": {}},
        "expected_checks": {"operation": "sum", "variable": "depth"},
        "category": "aggregation",
    },
    # --------------------------------------------------------------- SELECT
    {
        "id": "q15",
        "question": "Show me all observations from the Coral Sea",
        "expected_plan": {
            "operation": "select",
            "variable": None,
            "filters": {"region": "Coral Sea"},
        },
        "expected_checks": {"operation": "select", "region": "Coral Sea"},
        "category": "filter",
    },
    {
        "id": "q16",
        "question": "List the top 5 deepest observations",
        "expected_plan": {
            "operation": "select",
            "variable": "depth",
            "filters": {},
            "sort_by": {"column": "depth", "order": "desc"},
            "limit": 5,
        },
        "expected_checks": {"operation": "select"},
        "category": "intent",
    },
    # -------------------------------------------------------------- GROUP BY
    {
        "id": "q17",
        "question": "What is the average temperature by region?",
        "expected_plan": {
            "operation": "avg",
            "variable": "temperature",
            "filters": {},
            "group_by": ["region"],
        },
        "expected_checks": {"operation": "avg", "variable": "temperature", "group_by": ["region"]},
        "category": "aggregation",
    },
    {
        "id": "q18",
        "question": "Show me the count of observations by year",
        "expected_plan": {
            "operation": "count",
            "variable": "float_id",
            "filters": {},
            "group_by": ["year"],
        },
        "expected_checks": {"operation": "count", "group_by": ["year"]},
        "category": "aggregation",
    },
    {
        "id": "q19",
        "question": "Average salinity grouped by region",
        "expected_plan": {
            "operation": "avg",
            "variable": "salinity",
            "filters": {},
            "group_by": ["region"],
        },
        "expected_checks": {"operation": "avg", "variable": "salinity", "group_by": ["region"]},
        "category": "aggregation",
    },
    # --------------------------------------------------------------- DEPTH FILTER
    {
        "id": "q20",
        "question": "Show temperature observations shallower than 200 m",
        "expected_plan": {
            "operation": "select",
            "variable": "temperature",
            "filters": {"depth_range": {"min": None, "max": 200}},
        },
        "expected_checks": {"operation": "select", "has_depth_filter": True},
        "category": "filter",
    },
    {
        "id": "q21",
        "question": "Find average temperature below 500 m depth",
        "expected_plan": {
            "operation": "avg",
            "variable": "temperature",
            "filters": {"depth_range": {"min": 500, "max": None}},
        },
        "expected_checks": {"operation": "avg", "variable": "temperature", "has_depth_filter": True},
        "category": "filter",
    },
    # --------------------------------------------------------------- DATE FILTER
    {
        "id": "q22",
        "question": "Show all observations from 2021",
        "expected_plan": {
            "operation": "select",
            "variable": None,
            "filters": {"date_range": {"start": "2021-01-01", "end": "2021-12-31"}},
        },
        "expected_checks": {"operation": "select", "has_date_filter": True},
        "category": "filter",
    },
    {
        "id": "q23",
        "question": "What is the average temperature in 2022?",
        "expected_plan": {
            "operation": "avg",
            "variable": "temperature",
            "filters": {"date_range": {"start": "2022-01-01", "end": "2022-12-31"}},
        },
        "expected_checks": {"operation": "avg", "variable": "temperature", "has_date_filter": True},
        "category": "filter",
    },
    {
        "id": "q24",
        "question": "Count observations between 2020 and 2022",
        "expected_plan": {
            "operation": "count",
            "variable": "float_id",
            "filters": {"date_range": {"start": "2020-01-01", "end": "2022-12-31"}},
        },
        "expected_checks": {"operation": "count", "has_date_filter": True},
        "category": "filter",
    },
    # --------------------------------------------------------------- COMPARE
    {
        "id": "q25",
        "question": "Compare temperature between all ocean regions",
        "expected_plan": {
            "operation": "compare",
            "variable": "temperature",
            "filters": {},
        },
        "expected_checks": {"operation": "compare", "variable": "temperature"},
        "category": "aggregation",
    },
    # --------------------------------------------------------------- MULTI-FILTER
    {
        "id": "q26",
        "question": "Average salinity in the North Atlantic with good quality data only",
        "expected_plan": {
            "operation": "avg",
            "variable": "salinity",
            "filters": {"region": "North Atlantic", "qc_flag": [1, 2]},
        },
        "expected_checks": {"operation": "avg", "variable": "salinity", "region": "North Atlantic"},
        "category": "filter",
    },
    {
        "id": "q27",
        "question": "Find maximum depth in South Pacific observations from 2021",
        "expected_plan": {
            "operation": "max",
            "variable": "depth",
            "filters": {
                "region": "South Pacific",
                "date_range": {"start": "2021-01-01", "end": "2021-12-31"},
            },
        },
        "expected_checks": {
            "operation": "max",
            "variable": "depth",
            "region": "South Pacific",
            "has_date_filter": True,
        },
        "category": "filter",
    },
    {
        "id": "q28",
        "question": "Show shallow observations less than 100 m in the Caribbean Sea",
        "expected_plan": {
            "operation": "select",
            "variable": None,
            "filters": {
                "region": "Caribbean Sea",
                "depth_range": {"min": None, "max": 100},
            },
        },
        "expected_checks": {
            "operation": "select",
            "region": "Caribbean Sea",
            "has_depth_filter": True,
        },
        "category": "filter",
    },
    # --------------------------------------------------------------- RAG / DOMAIN
    {
        "id": "q29",
        "question": "What are Argo profiling floats?",
        "expected_plan": {"operation": "select", "variable": None, "filters": {}},
        "expected_checks": {"rag_docs_retrieved": True},
        "category": "rag",
    },
    {
        "id": "q30",
        "question": "Explain what QC flags mean in Argo data",
        "expected_plan": {"operation": "select", "variable": None, "filters": {}},
        "expected_checks": {"rag_docs_retrieved": True},
        "category": "rag",
    },
    {
        "id": "q31",
        "question": "What is the thermohaline circulation?",
        "expected_plan": {"operation": "select", "variable": None, "filters": {}},
        "expected_checks": {"rag_docs_retrieved": True},
        "category": "rag",
    },
    {
        "id": "q32",
        "question": "Show average pressure by QC flag",
        "expected_plan": {
            "operation": "avg",
            "variable": "pressure",
            "filters": {},
            "group_by": ["qc_flag"],
        },
        "expected_checks": {"operation": "avg", "variable": "pressure"},
        "category": "aggregation",
    },
]

