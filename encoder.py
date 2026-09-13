"""Semantic encoder interface: shared by all retrievers, swappable for offline tests."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class SemanticEncoder:
    """BGE / sentence-transformers encoder with L2 normalization and an in-memory cache.

    ``encode_batch`` returns an ``(N, dim)`` float32 matrix whose rows are already
    normalized (so the inner product equals cosine similarity). Tests may swap in
    a deterministic FakeEncoder.
    """

    DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

    def __init__(self, model_name: str | None = None, batch_size: int = 64):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.batch_size = batch_size
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            embeddings = self._model.encode(
                uncached,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, emb in zip(uncached, embeddings):
                self._cache[text] = emb
        return np.stack([self._cache[t] for t in texts])

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def clear_cache(self):
        self._cache.clear()


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    """L2-normalize each row (same as eval_rag_at5.normalize_rows)."""
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(n < 1e-8, 1.0, n)