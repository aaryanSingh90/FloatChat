"""
rag/retriever.py
================
High-level RAG retriever that wraps the KnowledgeBaseEmbedder and formats
retrieved context for downstream use (Gemini prompt or rule-based fallback).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Retrieves and formats relevant knowledge-base context for a user question.

    Parameters
    ----------
    embedder : KnowledgeBaseEmbedder
        A pre-built embedder with an active FAISS index.
    top_k : int
        Number of documents to retrieve per query.
    """

    def __init__(self, embedder, top_k: int = 3) -> None:
        self._embedder = embedder
        self._top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> list[dict]:
        """
        Return up to *top_k* relevant documents for *question*.
        Returns an empty list if the embedder is not ready.
        """
        if not self._embedder.is_ready:
            logger.warning("RAG retriever called but embedder is not ready.")
            return []
        try:
            docs = self._embedder.retrieve(question, top_k=self._top_k)
            return docs
        except Exception as exc:  # noqa: BLE001
            logger.error("RAG retrieval failed: %s", exc)
            return []

    def format_context(self, docs: list[dict]) -> str:
        """
        Format retrieved documents into a compact context string suitable
        for injection into an LLM prompt.
        """
        if not docs:
            return ""
        parts = []
        for i, doc in enumerate(docs, start=1):
            parts.append(
                f"[Doc {i}: {doc['title']}]\n{doc['content']}"
            )
        return "\n\n".join(parts)

    def retrieve_and_format(self, question: str) -> tuple[list[dict], str]:
        """Convenience method: retrieve and format in one call."""
        docs = self.retrieve(question)
        context = self.format_context(docs)
        return docs, context

    @property
    def top_k(self) -> int:
        return self._top_k

