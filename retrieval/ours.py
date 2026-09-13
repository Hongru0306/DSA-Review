"""Ours —— 超图子图检索(论文 Algorithm 1 + 2)。

忠实移植 ``scripts/eval_rag_at5.py`` ``OursCoverageGeneric``。对每条候选条款子图
``G_j``,以 ``omega_j(u)=idf(u)**p * gamma**hop(u)`` 加权做语义锚定聚合:

    N_e = Sum_t mean_top_N( M_Q[t,u] * omega_j(u) )          # 实体覆盖项
    N_r = Sum_rho max_(u,rho,v) (M_Q[a,u]omega+ M_Q[b,v]omega)/2  # 关系项
    Score = [ max(N_e + N_r - alpha*|V_j|, 0) ] / max(D_e + D_r, 1e-9)
    paper_score_formula=True 时: [ max(N_e - alpha*|V_j|, 0) + lambda*N_r ]
                                / max(D_e + lambda*D_r, 1e-9)

``D_e / D_r`` 为全局引用分(分母对全部候选恒定,不改变排序;两式排序等价)。
参数对齐论文参数敏感性网格最优 (N, alpha, lambda) = (3, 0.03, 0.85)。
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from config import MAX_REL
from encoder import normalize_rows
from graph.query_graph import resolve_query_graph
from retrieval.base import Retriever
from utils.text import doc_cache_key, extract_terms


def topk_mean(mat: np.ndarray, k: int) -> np.ndarray:
    """按行取 top-k 的均值;``k<=1`` 取 max,列数不足取全行均值。"""
    if mat.shape[1] == 0:
        return np.zeros(mat.shape[0], dtype=np.float32)
    if k <= 1:
        return mat.max(axis=1)
    if mat.shape[1] <= k:
        return mat.mean(axis=1)
    idx = np.argpartition(mat, -k, axis=1)[:, -k:]
    return np.take_along_axis(mat, idx, axis=1).mean(axis=1)


class OursRetriever(Retriever):
    name = "Ours"

    def __init__(
        self,
        n: int = 3,
        m: int = 2,
        gamma: float = 0.7,
        alpha: float = 0.03,
        lambda_: float = 0.85,
        idf_power: float = 0.25,
        top_k: int = 5,
        use_llm: bool = False,
        use_rel: bool = True,
        paper_score_formula: bool = True,
        use_semantic: bool = True,
        exp_cap: int = 300,
        local_cap: int = 512,
        global_edge_cap: int = 5000,
        q_cap: int = 10,
        name: str | None = None,
    ):
        self.n = int(n)          # 实体聚合大小(论文参数网格的 N)
        self.m = int(m)          # BFS 扩展跳数
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.lambda_ = float(lambda_)
        self.relation_weight = float(lambda_)
        self.idf_power = float(idf_power)
        self.top_k = int(top_k)  # 输出 top-K
        self.use_llm = use_llm
        self.use_rel = use_rel
        self.paper_score_formula = paper_score_formula
        self.use_semantic = use_semantic
        self.exp_cap = int(exp_cap)
        self.local_cap = int(local_cap)
        self.global_edge_cap = int(global_edge_cap)
        self.q_cap = int(q_cap)
        if name:
            self.name = name
        self.qcache: dict[str, dict[str, Any]] = {}
        self.corpus_graph: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "OursRetriever":
        return cls(**params)

    # ---- 附件接口(与实验代码兼容)----

    def attach_query_cache(self, cache: dict[str, dict[str, Any]] | None) -> None:
        self.qcache = cache or {}

    def attach_corpus_graph(self, graph: dict[str, dict[str, Any]] | None) -> None:
        self.corpus_graph = graph or {}

    # ---- 建索引 ----

    def build(
        self,
        corpus: list[str],
        encoder,
        corpus_graph: dict[str, dict[str, Any]] | None = None,
        qcache: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.encoder = encoder
        self.corpus = corpus
        self.n_docs = len(corpus)
        if corpus_graph is not None:
            self.corpus_graph = corpus_graph
        if qcache is not None:
            self.qcache = qcache

        key_to_name: dict[str, str] = {}
        self.doc_keys: list[list[str]] = []
        self.doc_rel_pairs: list[list[tuple[str, str]]] = []
        for text in corpus:
            graph_obj = getattr(self, "corpus_graph", {}).get(doc_cache_key(text), {})
            keys: list[str] = []
            for term in graph_obj.get("entities") or extract_terms(text, 64):
                k = term.lower()
                key_to_name.setdefault(k, term)
                keys.append(k)
            self.doc_keys.append(list(dict.fromkeys(keys)))
            rels = []
            for r in graph_obj.get("relations", []) or []:
                if len(r) >= 2:
                    rels.append((str(r[0]).lower(), str(r[1]).lower()))
            self.doc_rel_pairs.append(rels)

        self.entity_keys = list(key_to_name)
        self.entity_names = [key_to_name[k] for k in self.entity_keys]
        self.entity_idx = {k: i for i, k in enumerate(self.entity_keys)}
        self.entity_embs = (
            normalize_rows(encoder.encode_batch(self.entity_names).astype(np.float32))
            if self.entity_names
            else np.zeros((0, 512), dtype=np.float32)
        )

        df = np.zeros(len(self.entity_keys), dtype=np.float32)
        for keys in self.doc_keys:
            for k in keys:
                df[self.entity_idx[k]] += 1
        soft_idf = np.log(1 + self.n_docs / np.maximum(df, 0.5))
        self.idf_p = np.power(np.maximum(soft_idf, 1e-6), self.idf_power).astype(np.float32)

        adj: dict[int, set[int]] = defaultdict(set)
        self.edges: list[tuple[int, int]] = []
        for di, keys in enumerate(self.doc_keys):
            ids = [self.entity_idx[k] for k in keys if k in self.entity_idx]
            rel_edges = [
                (self.entity_idx[s], self.entity_idx[t])
                for s, t in self.doc_rel_pairs[di]
                if s in self.entity_idx and t in self.entity_idx
            ]
            if rel_edges:
                pairs = rel_edges
            else:
                pairs = []
                for i, u in enumerate(ids):
                    for v in ids[i + 1 : min(i + 8, len(ids))]:
                        pairs.append((u, v))
            for u, v in pairs:
                adj[u].add(v)
                adj[v].add(u)
                self.edges.append((u, v))
                self.edges.append((v, u))
        self.edge_adj = adj

        self.expansions: list[tuple[np.ndarray, np.ndarray]] = []
        self.local_edges_by_doc: list[list[tuple[int, int]]] = []
        for keys in self.doc_keys:
            seeds = [self.entity_idx[k] for k in keys if k in self.entity_idx]
            hopd: dict[int, int] = {i: 0 for i in seeds}
            frontier = deque((i, 0) for i in seeds)
            seen = set(seeds)
            while frontier:
                node, dep = frontier.popleft()
                if dep >= self.m:
                    continue
                for nb in adj.get(node, []):
                    hop = dep + 1
                    if hop < hopd.get(nb, 1 << 30):
                        hopd[nb] = hop
                    if nb not in seen:
                        seen.add(nb)
                        frontier.append((nb, hop))
            idxs = np.array(list(hopd), dtype=np.int32)
            hops = np.array([hopd[int(i)] for i in idxs], dtype=np.float32)
            w_idf = (self.gamma ** hops) * self.idf_p[idxs]
            if self.exp_cap and len(idxs) > self.exp_cap:
                keep = np.argpartition(w_idf, -self.exp_cap)[-self.exp_cap :]
                idxs, hops = idxs[keep], hops[keep]
            self.expansions.append((idxs, hops))

            local_edges: list[tuple[int, int]] = []
            for s, t in self.doc_rel_pairs[len(self.local_edges_by_doc)]:
                if s in self.entity_idx and t in self.entity_idx:
                    local_edges.append((int(self.entity_idx[s]), int(self.entity_idx[t])))
                    local_edges.append((int(self.entity_idx[t]), int(self.entity_idx[s])))
            if not local_edges:
                direct = [self.entity_idx[k] for k in keys if k in self.entity_idx][:24]
                for i, u in enumerate(direct):
                    for v in direct[i + 1 : min(i + 6, len(direct))]:
                        local_edges.append((int(u), int(v)))
                        local_edges.append((int(v), int(u)))
            self.local_edges_by_doc.append(local_edges[: self.local_cap])

        global_edges = (
            self.edges[: self.global_edge_cap]
            if len(self.edges) > self.global_edge_cap
            else self.edges
        )
        self.global_edge_us = np.array([u for u, _ in global_edges], dtype=np.int32)
        self.global_edge_vs = np.array([v for _, v in global_edges], dtype=np.int32)

        max_local_edges = max((len(x) for x in self.local_edges_by_doc), default=0)
        if max_local_edges:
            self.local_us_mat = np.zeros((self.n_docs, max_local_edges), dtype=np.int32)
            self.local_vs_mat = np.zeros((self.n_docs, max_local_edges), dtype=np.int32)
            self.local_edge_mask = np.zeros((self.n_docs, max_local_edges), dtype=bool)
            for di, local_edges in enumerate(self.local_edges_by_doc):
                if not local_edges:
                    continue
                us = [u for u, _ in local_edges]
                vs = [v for _, v in local_edges]
                n = len(local_edges)
                self.local_us_mat[di, :n] = us
                self.local_vs_mat[di, :n] = vs
                self.local_edge_mask[di, :n] = True
        else:
            self.local_us_mat = np.zeros((self.n_docs, 0), dtype=np.int32)
            self.local_vs_mat = np.zeros((self.n_docs, 0), dtype=np.int32)
            self.local_edge_mask = np.zeros((self.n_docs, 0), dtype=bool)

        max_len = max((len(idxs) for idxs, _ in self.expansions), default=0)
        self.exp_lens = np.array([len(idxs) for idxs, _ in self.expansions], dtype=np.int32)
        if max_len:
            self.exp_idx_mat = np.zeros((self.n_docs, max_len), dtype=np.int32)
            self.exp_hop_mat = np.zeros((self.n_docs, max_len), dtype=np.float32)
            self.exp_mask = np.zeros((self.n_docs, max_len), dtype=bool)
            for di, (idxs, hops) in enumerate(self.expansions):
                n = len(idxs)
                self.exp_idx_mat[di, :n] = idxs
                self.exp_hop_mat[di, :n] = hops
                self.exp_mask[di, :n] = True
        else:
            self.exp_idx_mat = np.zeros((self.n_docs, 0), dtype=np.int32)
            self.exp_hop_mat = np.zeros((self.n_docs, 0), dtype=np.float32)
            self.exp_mask = np.zeros((self.n_docs, 0), dtype=bool)
        self.set_gamma(self.gamma)

    def set_gamma(self, gamma: float) -> None:
        self.gamma = gamma
        if getattr(self, "exp_idx_mat", None) is not None and self.exp_idx_mat.shape[1]:
            w = (gamma**self.exp_hop_mat) * self.idf_p[self.exp_idx_mat]
            self.exp_w_mat = np.where(self.exp_mask, w, 0.0).astype(np.float32)
        else:
            self.exp_w_mat = np.zeros((self.n_docs, 0), dtype=np.float32)

    # ---- 查询 ----

    def query_terms_rels(self, q: str) -> tuple[list[str], list[list[str]]]:
        return resolve_query_graph(q, self.qcache, use_llm=self.use_llm)

    # ---- 打分 ----

    def score(self, q: str) -> np.ndarray:
        q_terms, q_rels = self.query_terms_rels(q)
        if self.q_cap and len(q_terms) > self.q_cap:
            q_terms = sorted(q_terms, key=len, reverse=True)[: self.q_cap]
        if not q_terms or len(self.entity_names) == 0:
            return np.zeros(self.n_docs, dtype=np.float32)

        q_embs = normalize_rows(self.encoder.encode_batch(q_terms).astype(np.float32))
        q_sims = q_embs @ self.entity_embs.T
        if not self.use_semantic:
            bm = np.zeros_like(q_sims)
            for i, term in enumerate(q_terms):
                j = self.entity_idx.get(str(term).strip().lower())
                if j is not None:
                    bm[i, j] = 1.0
            q_sims = bm

        w_A = float(topk_mean(q_sims * self.idf_p, self.n).sum())
        rel_den = 0.0
        rel_vecs: list[tuple[np.ndarray, np.ndarray]] = []
        if self.use_rel and q_rels and self.edges:
            terms = [x for r in q_rels[:MAX_REL] for x in r[:2]]
            raw = normalize_rows(self.encoder.encode_batch(terms).astype(np.float32))
            for i, _rel in enumerate(q_rels[:MAX_REL]):
                e1, e2 = raw[2 * i], raw[2 * i + 1]
                us, vs = self.global_edge_us, self.global_edge_vs
                pair = (
                    self.entity_embs[us] @ e1 * self.idf_p[us]
                    + self.entity_embs[vs] @ e2 * self.idf_p[vs]
                ) / 2
                rel_den += float(pair.max()) if len(pair) else 0.0
                rel_vecs.append((e1, e2))

        inter_vec = np.zeros(self.n_docs, dtype=np.float32)
        if self.exp_idx_mat.shape[1]:
            denom = np.minimum(np.maximum(self.exp_lens, 1), self.n).astype(np.float32)
            for row in q_sims:
                vals = row[self.exp_idx_mat] * self.exp_w_mat
                vals = np.where(self.exp_mask, vals, -np.inf)
                if vals.shape[1] <= self.n:
                    top = vals
                else:
                    idx = np.argpartition(vals, -self.n, axis=1)[:, -self.n :]
                    top = np.take_along_axis(vals, idx, axis=1)
                top = np.where(np.isfinite(top), top, 0.0)
                inter_vec += top.sum(axis=1) / denom

        rel_num_vec = np.zeros(self.n_docs, dtype=np.float32)
        if self.use_rel and rel_vecs and self.local_us_mat.shape[1]:
            for e1, e2 in rel_vecs:
                u_scores = (self.entity_embs @ e1) * self.idf_p
                v_scores = (self.entity_embs @ e2) * self.idf_p
                pair = (u_scores[self.local_us_mat] + v_scores[self.local_vs_mat]) / 2
                pair = np.where(self.local_edge_mask, pair, -np.inf)
                rel_num_vec += np.where(
                    np.isfinite(pair.max(axis=1)), pair.max(axis=1), 0.0
                ).astype(np.float32)

        doc_key_lens = np.array([len(x) for x in self.doc_keys], dtype=np.float32)
        if self.paper_score_formula:
            entity_score = np.maximum(inter_vec - self.alpha * doc_key_lens, 0.0)
            scores = (entity_score + self.relation_weight * rel_num_vec) / max(
                w_A + self.relation_weight * rel_den, 1e-9
            )
        else:
            scores = np.maximum(
                inter_vec + rel_num_vec - self.alpha * doc_key_lens, 0.0
            ) / max(w_A + rel_den, 1e-9)
        return scores

    def rank(self, q: str, return_scores: bool = False):
        q_terms = self.query_terms_rels(q)[0]
        if self.q_cap and len(q_terms) > self.q_cap:
            q_terms = sorted(q_terms, key=len, reverse=True)[: self.q_cap]
        if not q_terms or len(self.entity_names) == 0:
            if return_scores:
                return np.zeros(self.n_docs, dtype=np.float32)
            return list(range(self.n_docs))
        scores = self.score(q)
        if return_scores:
            return scores
        return list(np.argsort(scores)[::-1])

    def retrieve(self, q: str, top_k: int | None = None) -> list[int]:
        return self.rank(q)[: (top_k if top_k is not None else self.top_k)]