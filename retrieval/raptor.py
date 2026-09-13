"""RAPTOR-lite 基线(局部抽取式层级摘要检索)。

移植 ``scripts/eval_rag_at5.py`` RAPTORLite:按 8 文档连续窗口分簇,摘要 = 各成员
前 350 字符拼接;查询对叶节点 + 簇双路稠密打分,top-8 簇给成员文档
``0.15 * c_score`` 加成。
"""

from __future__ import annotations

from typing import List

import numpy as np

from retrieval.base import Retriever


class RAPTORLite(Retriever):
    """RAPTOR-style hierarchical summary retrieval using local extractive clusters."""

    name = "RAPTOR-lite"
    CLUSTER_SIZE = 8
    CLUSTER_BOOST = 0.15
    TOP_CLUSTERS = 8

    def build(self, corpus: List[str], encoder) -> None:
        self.corpus, self.encoder = corpus, encoder
        self.doc_embs = encoder.encode_batch(corpus)
        self.clusters = []
        for start in range(0, len(corpus), self.CLUSTER_SIZE):
            ids = list(range(start, min(start + self.CLUSTER_SIZE, len(corpus))))
            summary = " ".join(corpus[i][:350] for i in ids)
            self.clusters.append((ids, summary))
        self.cluster_embs = (
            encoder.encode_batch([s for _, s in self.clusters])
            if self.clusters
            else np.zeros((0, self.doc_embs.shape[1]))
        )

    def rank(self, q: str) -> List[int]:
        qe = self.encoder.encode_batch([q])[0]
        c_scores = self.cluster_embs @ qe if len(self.clusters) else np.zeros(0)
        scores = self.doc_embs @ qe
        boosted = scores.copy()
        for ci in np.argsort(c_scores)[::-1][: max(1, min(self.TOP_CLUSTERS, len(self.clusters)))]:
            for di in self.clusters[int(ci)][0]:
                boosted[di] += self.CLUSTER_BOOST * c_scores[int(ci)]
        return list(np.argsort(boosted)[::-1])