"""词项共现图 + Personalized PageRank 检索基线(GraphRAG-lite / HippoRAG-lite)。

移植 ``scripts/eval_rag_at5.py`` GraphRAGLite / HippoRAGLite:词项图(窗口 10 共现,
行归一邻接)上跑受限步数 PPR,再按词项 IDF 累积到文档分。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List

import numpy as np

from retrieval.base import Retriever
from utils.text import extract_terms


class GraphRAGLite(Retriever):
    name = "GraphRAG-lite"

    def __init__(self, name: str | None = None):
        if name:
            self.name = name

    def build(self, corpus: List[str], encoder) -> None:
        self.corpus = corpus
        self.doc_terms = [extract_terms(d, 64) for d in corpus]
        self.term2id: dict[str, int] = {}
        self.term_docs: dict[int, set[int]] = defaultdict(set)
        edge_w: dict[tuple[int, int], float] = defaultdict(float)
        for di, terms in enumerate(self.doc_terms):
            for term in terms:
                k = term.lower()
                self.term2id.setdefault(k, len(self.term2id))
                self.term_docs[self.term2id[k]].add(di)
        n = len(self.term2id)
        for terms in self.doc_terms:
            ids = [self.term2id[t.lower()] for t in terms if t.lower() in self.term2id]
            for i, u in enumerate(ids):
                for v in ids[i + 1 : min(i + 10, len(ids))]:
                    edge_w[(u, v)] += 1.0
                    edge_w[(v, u)] += 1.0
        rows: dict[int, list[tuple[int, float]]] = defaultdict(list)
        row_sum: dict[int, float] = defaultdict(float)
        for (u, v), w in edge_w.items():
            rows[u].append((v, w))
            row_sum[u] += w
        self.adj = {
            u: [(v, w / row_sum[u]) for v, w in vs]
            for u, vs in rows.items()
            if row_sum[u] > 0
        }
        self.df = np.array([len(self.term_docs[i]) for i in range(n)], dtype=np.float32)
        self.n_terms, self.n_docs = n, len(corpus)

    def _ppr(
        self,
        seeds: List[int],
        walk: float = 0.50,
        restart_w: float = 0.20,
        steps: int = 2,
        max_active: int = 800,
    ) -> dict[int, float]:
        seeds = [int(s) for s in seeds][:256]
        if not seeds:
            return {}
        p: dict[int, float] = {s: 1.0 / len(seeds) for s in seeds}
        frontier = dict(p)
        decay = walk
        max_active = min(int(max_active), 1500)
        for _ in range(min(int(steps), 6)):
            nxt: dict[int, float] = {}
            for u, pu in frontier.items():
                adj_u = self.adj.get(u, ())
                if not adj_u:
                    continue
                for pair in adj_u:
                    try:
                        v, w = int(pair[0]), float(pair[1])
                    except Exception:
                        continue
                    val = decay * pu * w
                    if val > 0:
                        nxt[v] = nxt.get(v, 0.0) + val
            if len(nxt) > max_active:
                nxt = dict(sorted(nxt.items(), key=lambda x: x[1], reverse=True)[:max_active])
            for u, val in nxt.items():
                if val > p.get(u, 0.0):
                    p[u] = val
            frontier = nxt
            decay *= walk
            if not frontier:
                break
        return p

    def rank(self, q: str) -> List[int]:
        seeds = [
            self.term2id[t.lower()]
            for t in extract_terms(q, 48)
            if t.lower() in self.term2id
        ]
        if not seeds or not self.n_terms:
            return list(range(self.n_docs))
        p = self._ppr(seeds, 0.50, 0.20, 2, 800)
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for tid, val in p.items():
            val *= math.log(1 + self.n_docs / max(float(self.df[tid]), 1.0))
            if val > 0:
                for di in self.term_docs.get(tid, ()):
                    scores[di] += val
        return list(np.argsort(scores)[::-1])


class HippoRAGLite(GraphRAGLite):
    """与 GraphRAGLite 同构,仅 PPR 参数与 IDF 加权不同。"""

    name = "HippoRAG-lite"

    def rank(self, q: str) -> List[int]:
        seeds = [
            self.term2id[t.lower()]
            for t in extract_terms(q, 48)
            if t.lower() in self.term2id
        ]
        if not seeds or not self.n_terms:
            return list(range(self.n_docs))
        p = self._ppr(seeds, 0.55, 0.15, 3, 1200)
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for tid, val in p.items():
            if val > 0:
                for di in self.term_docs.get(tid, ()):
                    scores[di] += val
        return list(np.argsort(scores)[::-1])