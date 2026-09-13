"""LightRAG-lite 基线(图无关,词项 IDF 加权重叠)。

移植 ``scripts/eval_rag_at5.py`` LightRAGLite:
``score = Sum(ln(1 + n/df(t)) for t in overlap) + |qt∩ts| / |qt∪ts|``。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List

import numpy as np

from retrieval.base import Retriever
from utils.text import extract_terms


class LightRAGLite(Retriever):
    name = "LightRAG-lite"

    def build(self, corpus: List[str], encoder) -> None:
        self.terms = [set(t.lower() for t in extract_terms(d, 64)) for d in corpus]
        self.df = Counter(t for ts in self.terms for t in ts)
        self.n = len(corpus)

    def rank(self, q: str) -> List[int]:
        qt = set(t.lower() for t in extract_terms(q, 48))
        scores = []
        for ts in self.terms:
            ov = qt & ts
            scores.append(
                sum(math.log(1 + self.n / max(self.df[t], 1)) for t in ov)
                + len(ov) / max(len(qt | ts), 1)
            )
        return list(np.argsort(scores)[::-1])