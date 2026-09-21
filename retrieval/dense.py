"""Dense BGE retrieval baseline (NaiveRAG).

Port of ``scripts/eval_rag_at5.py`` DenseNaive: L2-normalized embeddings with a
full-corpus cosine scan.
"""

from __future__ import annotations

from typing import List

import numpy as np

from retrieval.base import Retriever


class DenseNaive(Retriever):
    name = "NaiveRAG"

    def __init__(self, name: str | None = None):
        if name:
            self.name = name

    def build(self, corpus: List[str], encoder) -> None:
        self.encoder, self.embs = encoder, encoder.encode_batch(corpus)

    def rank(self, q: str) -> List[int]:
        qe = self.encoder.encode_batch([q])[0]
        return [int(i) for i in np.argsort(self.embs @ qe)[::-1]]