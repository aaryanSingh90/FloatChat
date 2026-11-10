"""
tests/test_rag.py
==================
Tests for the RAG embedder and retriever.
Skipped automatically if sentence-transformers or faiss-cpu are not installed.
"""

from __future__ import annotations

import pytest

# Skip entire module if deps are missing
pytest.importorskip("faiss", reason="faiss-cpu not installed")
pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")

from rag.embedder import KnowledgeBaseEmbedder
from rag.retriever import RAGRetriever
from knowledge_base.documents import DOCUMENTS


MINI_DOCS = DOCUMENTS[:5]  # Use a subset for speed


@pytest.fixture(scope="module")
def built_embedder():
    embedder = KnowledgeBaseEmbedder()
    embedder.build_index(MINI_DOCS)
    return embedder


class TestEmbedder:
    def test_builds_successfully(self, built_embedder):
        assert built_embedder.is_ready

    def test_document_count(self, built_embedder):
        assert built_embedder.document_count == len(MINI_DOCS)

    def test_retrieve_returns_docs(self, built_embedder):
        docs = built_embedder.retrieve("What are Argo floats?", top_k=2)
        assert len(docs) == 2
        assert "title" in docs[0]
        assert "content" in docs[0]
        assert "score" in docs[0]

    def test_retrieve_top_k_respected(self, built_embedder):
        docs = built_embedder.retrieve("temperature", top_k=1)
        assert len(docs) == 1

    def test_argo_query_retrieves_argo_doc(self, built_embedder):
        docs = built_embedder.retrieve("Argo profiling float cycle", top_k=1)
        assert docs[0]["id"].startswith("argo")

    def test_embed_query_returns_vector(self, built_embedder):
        import numpy as np
        vec = built_embedder.embed_query("salinity oceanography")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1

    def test_not_ready_before_build(self):
        fresh = KnowledgeBaseEmbedder()
        assert not fresh.is_ready

    def test_retrieve_raises_if_not_ready(self):
        fresh = KnowledgeBaseEmbedder()
        with pytest.raises(RuntimeError, match="not built"):
            fresh.retrieve("test")


class TestRetriever:
    def test_retrieve_and_format(self, built_embedder):
        retriever = RAGRetriever(built_embedder, top_k=2)
        docs, context = retriever.retrieve_and_format("QC flag quality control")
        assert len(docs) >= 1
        assert len(context) > 0
        assert "[Doc 1:" in context

    def test_empty_result_on_unready_embedder(self):
        fresh = KnowledgeBaseEmbedder()
        retriever = RAGRetriever(fresh, top_k=3)
        docs = retriever.retrieve("test")
        assert docs == []

    def test_format_empty_docs(self, built_embedder):
        retriever = RAGRetriever(built_embedder)
        context = retriever.format_context([])
        assert context == ""

