"""
config/settings.py
==================
Centralised configuration for Float Chat.
Values are read from environment variables (and .env if python-dotenv is
installed), with sensible defaults so the application runs without any
environment setup.
"""

from __future__ import annotations

import os
import logging

# ---------------------------------------------------------------------------
# Optional .env loading
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv not installed; env vars still work normally

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ---------------------------------------------------------------------------
# Query settings
# ---------------------------------------------------------------------------
QUERY_MAX_ROWS: int = int(os.getenv("QUERY_MAX_ROWS", "100"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# RAG settings
# ---------------------------------------------------------------------------
RAG_EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))

# ---------------------------------------------------------------------------
# Dataset path
# ---------------------------------------------------------------------------
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "sample_argo_data.csv"

