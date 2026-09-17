"""Okapi BM25 retrieval baseline.

Port of ``scripts/eval_rag_at5.py`` BM25: k1=1.2 / b=0.75,
``idf = ln(1 + (N-df+0.5)/(df+0.5))``,
weight ``idf * tf * 2.2 / (tf + k1*(1-b+b*dl/avgdl))``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

import numpy as np

from config import STOP_EN, STOP_ZH
from retrieval.base import Retriever

_TOKEN_RE = re.compile(r"[A-Za-z0-9一-鿿][A-Za-z0-9一-鿿'’./-]*")


class BM25(Retriever):
    name = "BM25"

    def __init__(self, k1: float = 1.2, b: float = 0.75, name: str | None = None):
        self.k1 = k1
        self.b = b
        if name:
            self.name = name

    @staticmethod
    def _tok(text) -> List[str]:
        return [
            w.lower()
            for w in _TOKEN_RE.findall(str(text))
            if w.lower() not in STOP_EN and w not in STOP_ZH
        ]

    def build(self, corpus: List[str], encoder) -> None:
        self.docs = [self._tok(d) for d in corpus]
        self.n = len(self.docs)
        self.df = Counter(t for d in self.docs for t in set(d))
        self.avgdl = sum(len(d) for d in self.docs) / max(self.n, 1)

    def rank(self, q: str) -> List[int]:
        qt = self._tok(q)
        out = []
        for d in self.docs:
            tf = Counter(d)
            score = 0.0
            for t in qt:
                if t not in tf:
                    continue
                df = self.df.get(t, 0)
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                denom = tf[t] + self.k1 * (
                    1 - self.b + self.b * len(d) / max(self.avgdl, 1e-9)
                )
                score += idf * tf[t] * (self.k1 + 1) / denom
            out.append(score)
        return list(np.argsort(out)[::-1])