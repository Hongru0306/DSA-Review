"""语义编码器接口:所有检索器共享,便于替换/注入假编码器做离线测试。"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class SemanticEncoder:
    """BGE 等 sentence-transformers 编码器,L2 归一化 + 内存缓存。

    ``encode_batch`` 返回 ``(N, dim)`` 的 float32 矩阵,每行已归一化
    (即内积即余弦相似度)。测试时可替换为确定性 FakeEncoder。
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
    """对行做 L2 归一化(与 eval_rag_at5.normalize_rows 一致)。"""
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(n < 1e-8, 1.0, n)