"""
rag/embedder.py
===============
Builds and manages a FAISS vector index over the scientific knowledge base.
Uses Sentence Transformers for embedding (runs entirely locally).
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports so the module can be imported even if heavy deps are absent
_faiss = None
_SentenceTransformer = None


def _load_deps():
    global _faiss, _SentenceTransformer
    if _faiss is None:
        import faiss as _f
        _faiss = _f
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as _ST
        _SentenceTransformer = _ST


class KnowledgeBaseEmbedder:
    """
    Embeds the knowledge-base documents and stores them in a FAISS index.

    Parameters
    ----------
    model_name : str
        HuggingFace Sentence Transformers model identifier.
        'all-MiniLM-L6-v2' is a good balance of speed and quality.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Optional[object] = None
        self._index: Optional[object] = None          # faiss.Index
        self._documents: list[dict] = []
        self._embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self, documents: list[dict]) -> None:
        """
        Embed *documents* and build the FAISS index.

        Each document must have at least 'id', 'title', and 'content' keys.
        """
        _load_deps()
        self._documents = documents

        logger.info("Loading embedding model '%s' …", self._model_name)
        self._model = _SentenceTransformer(self._model_name)

        texts = [f"{doc['title']}\n{doc['content']}" for doc in documents]
        logger.info("Embedding %d documents …", len(texts))
        self._embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        dim = self._embeddings.shape[1]
        self._index = _faiss.IndexFlatL2(dim)
        self._index.add(self._embeddings.astype(np.float32))
        logger.info("FAISS index built with %d vectors (dim=%d).", len(texts), dim)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve the *top_k* most relevant documents for *query*.

        Returns a list of document dicts augmented with a 'score' key.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        _load_deps()
        q_emb = self._model.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self._index.search(q_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = dict(self._documents[idx])
            doc["score"] = float(dist)          # L2 distance (lower = more similar)
            results.append(doc)

        return results

    def embed_query(self, query: str) -> np.ndarray:
        """Return the embedding vector for a single query string."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call build_index() first.")
        return self._model.encode([query], convert_to_numpy=True)[0]

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._documents) > 0

    @property
    def document_count(self) -> int:
        return len(self._documents)

