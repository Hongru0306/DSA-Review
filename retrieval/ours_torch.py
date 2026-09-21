"""Torch-vectorized scoring for Ours retrieval (optional CUDA).

Port of ``scripts/ours_torch_accel.py``: construction and BGE embeddings stay in
numpy; only the scoring arithmetic is mirrored onto torch tensors. After
``to_torch(device)``, ``rank`` is computed by torch. Without an explicit
``to_torch`` call it is equivalent to the parent numpy implementation.
"""

from __future__ import annotations

import numpy as np

from config import MAX_REL
from encoder import normalize_rows
from retrieval.ours import OursRetriever


class OursRetrieverTorch(OursRetriever):
    def to_torch(self, device: str | None = None) -> "OursRetrieverTorch":
        import torch

        self._torch = torch
        self._torch_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dev = self._torch_device
        self.entity_embs_t = torch.as_tensor(self.entity_embs, dtype=torch.float32, device=dev)
        self.idf_p_t = torch.as_tensor(self.idf_p, dtype=torch.float32, device=dev)
        self.exp_idx_mat_t = torch.as_tensor(self.exp_idx_mat, dtype=torch.long, device=dev)
        self.exp_w_mat_t = torch.as_tensor(self.exp_w_mat, dtype=torch.float32, device=dev)
        self.exp_mask_t = torch.as_tensor(self.exp_mask, dtype=torch.bool, device=dev)
        self.exp_lens_t = torch.as_tensor(self.exp_lens, dtype=torch.float32, device=dev)
        self.local_us_mat_t = torch.as_tensor(self.local_us_mat, dtype=torch.long, device=dev)
        self.local_vs_mat_t = torch.as_tensor(self.local_vs_mat, dtype=torch.long, device=dev)
        self.local_edge_mask_t = torch.as_tensor(self.local_edge_mask, dtype=torch.bool, device=dev)
        self.global_edge_us_t = torch.as_tensor(self.global_edge_us, dtype=torch.long, device=dev)
        self.global_edge_vs_t = torch.as_tensor(self.global_edge_vs, dtype=torch.long, device=dev)
        self.doc_key_lens_t = torch.as_tensor(
            [len(x) for x in self.doc_keys], dtype=torch.float32, device=dev
        )
        return self

    def set_gamma(self, gamma: float) -> None:
        super().set_gamma(gamma)
        if getattr(self, "_torch", None) is not None:
            self.exp_w_mat_t = self._torch.as_tensor(
                self.exp_w_mat, dtype=torch.float32, device=self._torch_device
            )

    def rank(self, q: str, return_scores: bool = False):
        if getattr(self, "_torch", None) is None:
            return super().rank(q, return_scores=return_scores)
        scores = self.score_torch(q)
        if return_scores:
            return scores.detach().cpu().numpy().astype(np.float32)
        order = self._torch.argsort(scores, descending=True).detach().cpu().numpy()
        return [int(x) for x in order]

    def score_torch(self, q: str):
        torch = self._torch
        dev = self._torch_device

        q_terms, q_rels = self.query_terms_rels(q)
        if self.q_cap and len(q_terms) > self.q_cap:
            q_terms = sorted(q_terms, key=len, reverse=True)[: self.q_cap]
        if not q_terms or len(self.entity_names) == 0:
            return torch.zeros(self.n_docs, dtype=torch.float32, device=dev)

        q_np = normalize_rows(self.encoder.encode_batch(q_terms).astype(np.float32))
        q_embs = torch.as_tensor(q_np, dtype=torch.float32, device=dev)
        q_sims = q_embs @ self.entity_embs_t.T

        if not self.use_semantic:
            bm = torch.zeros_like(q_sims)
            for i, term in enumerate(q_terms):
                j = self.entity_idx.get(str(term).strip().lower())
                if j is not None:
                    bm[i, j] = 1.0
            q_sims = bm

        k = int(self.n)
        q_weighted = q_sims * self.idf_p_t
        if k <= 1:
            w_A = q_weighted.max(dim=1).values.sum()
        elif q_weighted.shape[1] <= k:
            w_A = q_weighted.mean(dim=1).sum()
        else:
            w_A = torch.topk(q_weighted, k, dim=1).values.mean(dim=1).sum()

        rel_den = torch.tensor(0.0, dtype=torch.float32, device=dev)
        rel_vecs: list[tuple] = []
        if self.use_rel and q_rels and len(self.edges):
            terms = [x for r in q_rels[:MAX_REL] for x in r[:2]]
            raw_np = normalize_rows(self.encoder.encode_batch(terms).astype(np.float32))
            raw = torch.as_tensor(raw_np, dtype=torch.float32, device=dev)
            for i, _rel in enumerate(q_rels[:MAX_REL]):
                e1, e2 = raw[2 * i], raw[2 * i + 1]
                us, vs = self.global_edge_us_t, self.global_edge_vs_t
                if us.numel():
                    pair = (
                        (self.entity_embs_t[us] @ e1) * self.idf_p_t[us]
                        + (self.entity_embs_t[vs] @ e2) * self.idf_p_t[vs]
                    ) / 2
                    rel_den = rel_den + pair.max()
                rel_vecs.append((e1, e2))

        if self.exp_idx_mat_t.shape[1]:
            denom = torch.minimum(
                torch.clamp(self.exp_lens_t, min=1.0), torch.tensor(float(k), device=dev)
            )
            vals = q_sims[:, self.exp_idx_mat_t] * self.exp_w_mat_t.unsqueeze(0)
            vals = vals.masked_fill(~self.exp_mask_t.unsqueeze(0), float("-inf"))
            if vals.shape[2] <= k:
                top = vals
            else:
                top = torch.topk(vals, k, dim=2).values
            top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
            inter_vec = top.sum(dim=2).sum(dim=0) / denom
        else:
            inter_vec = torch.zeros(self.n_docs, dtype=torch.float32, device=dev)

        rel_num_vec = torch.zeros(self.n_docs, dtype=torch.float32, device=dev)
        if self.use_rel and rel_vecs and self.local_us_mat_t.shape[1]:
            for e1, e2 in rel_vecs:
                u_scores = (self.entity_embs_t @ e1) * self.idf_p_t
                v_scores = (self.entity_embs_t @ e2) * self.idf_p_t
                pair = (u_scores[self.local_us_mat_t] + v_scores[self.local_vs_mat_t]) / 2
                pair = pair.masked_fill(~self.local_edge_mask_t, float("-inf"))
                best = pair.max(dim=1).values
                rel_num_vec = rel_num_vec + torch.where(
                    torch.isfinite(best), best, torch.zeros_like(best)
                )

        if self.paper_score_formula:
            entity_score = torch.clamp(
                inter_vec - float(self.alpha) * self.doc_key_lens_t, min=0.0
            )
            scores = entity_score + float(self.relation_weight) * rel_num_vec
            scores = scores / torch.clamp(
                w_A + float(self.relation_weight) * rel_den, min=1e-9
            )
        else:
            scores = torch.clamp(
                inter_vec + rel_num_vec - float(self.alpha) * self.doc_key_lens_t,
                min=0.0,
            )
            scores = scores / torch.clamp(w_A + rel_den, min=1e-9)
        return scores