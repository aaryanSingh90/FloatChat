"""
ai/pipeline.py
==============
Integrates the RAG retriever, Gemini client, and rule-based fallback into
a single end-to-end pipeline for Float Chat.

Usage
-----
    pipeline = AIPipeline(embedder=embedder, gemini=gemini_client)
    result = pipeline.process(question, df)
    # result.query_plan  → validated plan
    # result.rag_docs    → retrieved documents
    # result.mode        → "gemini" | "rule_based"
    # result.query_result → dict from query engine
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from ai.intent_parser import parse_intent
from core.query_engine import QueryEngine

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Holds every artefact produced during a single question's processing."""
    question: str
    mode: str                           # "gemini" or "rule_based"
    rag_docs: list[dict] = field(default_factory=list)
    rag_context: str = ""
    raw_plan: dict[str, Any] = field(default_factory=dict)
    query_plan: dict[str, Any] = field(default_factory=dict)
    query_result: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class AIPipeline:
    """
    Orchestrates: RAG retrieval → intent parsing → query execution.

    Parameters
    ----------
    embedder : KnowledgeBaseEmbedder | None
        Pre-built FAISS embedder.  If None, RAG is skipped.
    gemini : GeminiClient | None
        Gemini client.  If None or unavailable, falls back to rule-based.
    top_k : int
        Number of RAG documents to retrieve.
    """

    def __init__(self, embedder=None, gemini=None, top_k: int = 3) -> None:
        self._embedder = embedder
        self._gemini = gemini
        self._top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, question: str, df: pd.DataFrame) -> PipelineResult:
        """Full pipeline: RAG → intent → validate → execute."""
        result = PipelineResult(question=question, mode="rule_based")

        # 1. RAG retrieval
        if self._embedder and self._embedder.is_ready:
            try:
                docs = self._embedder.retrieve(question, top_k=self._top_k)
                result.rag_docs = docs
                result.rag_context = _format_context(docs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("RAG retrieval failed: %s", exc)

        # 2. Intent parsing (Gemini or rule-based)
        raw_plan = self._parse_intent(question, result.rag_context, result)

        result.raw_plan = raw_plan

        # 3. Query execution via QueryEngine (handles validation + optimisation)
        engine = QueryEngine(df)
        query_result = engine.run(raw_plan)
        result.query_plan = query_result.get("plan", raw_plan)
        result.query_result = query_result
        if query_result.get("error"):
            result.error = query_result["error"]

        return result

    @property
    def uses_gemini(self) -> bool:
        return bool(self._gemini and self._gemini.is_available)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _parse_intent(
        self, question: str, rag_context: str, result: PipelineResult
    ) -> dict[str, Any]:
        if self._gemini and self._gemini.is_available:
            try:
                plan = self._gemini.build_query_plan(question, rag_context)
                result.mode = "gemini"
                return plan
            except Exception as exc:  # noqa: BLE001
                logger.warning("Gemini call failed (%s), falling back to rule-based.", exc)

        result.mode = "rule_based"
        return parse_intent(question)


def _format_context(docs: list[dict]) -> str:
    if not docs:
        return ""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[Doc {i}: {doc['title']}]\n{doc['content']}")
    return "\n\n".join(parts)

