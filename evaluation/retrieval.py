"""Retrieval metrics: Hit@k / Recall / Precision / MAP / MRR / nDCG / coverage.

Two conventions (ported to match the experiment code):
- ``retrieval_metrics``: GraphRAG-Bench convention, ``dcg = Sum rel_i/log2(i+2)``.
- ``review_retrieval_metrics``: review-pipeline convention, ``dcg = Sum 1/log2(rank+1)``,
  plus evidence coverage and spec hits.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def retrieval_metrics(
    ranked: Iterable[int],
    gold: Iterable[int],
    k: int = 5,
) -> dict[str, float]:
    """Port matching ``scripts/eval_rag_at5.py`` retrieval_metrics."""
    gold_set = set(gold)
    top = list(ranked[:k])
    first = next((i + 1 for i, d in enumerate(ranked) if d in gold_set), 0)
    hits = sum(1 for d in top if d in gold_set)
    dcg = sum((1.0 if d in gold_set else 0.0) / math.log2(i + 2) for i, d in enumerate(top))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), k)))
    ap_num, ap_den = 0.0, min(len(gold_set), k)
    seen = 0
    for i, d in enumerate(top, 1):
        if d in gold_set:
            seen += 1
            ap_num += seen / i
    return {
        "MRR@k": 1.0 / first if first and first <= k else 0.0,
        "Hit@k": 1.0 if hits else 0.0,
        "Recall@k": hits / max(len(gold_set), 1),
        "Precision@k": hits / k,
        "nDCG@k": dcg / ideal if ideal else 0.0,
        "MAP@k": ap_num / ap_den if ap_den else 0.0,
    }


def review_retrieval_metrics(
    gold_values: Iterable[int],
    ranking: Iterable[int],
    k: int,
    *,
    gold_specs: Iterable[str] = (),
    ranked_specs: Iterable[str] = (),
) -> dict[str, float]:
    """Port matching ``scripts/run_construction_review_v2.py`` retrieval_metrics."""
    unique: list[int] = []
    seen: set[int] = set()
    for value in gold_values:
        doc_id = int(value)
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    gold_set = set(unique)
    top = [int(value) for value in list(ranking)[:k]]
    hit = 1.0 if gold_set & set(top) else 0.0
    coverage = len(gold_set & set(top)) / len(gold_set) if gold_set else 0.0
    reciprocal = 0.0
    for rank, doc_id in enumerate(top, 1):
        if doc_id in gold_set:
            reciprocal = 1.0 / rank
            break
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, doc_id in enumerate(top, 1) if doc_id in gold_set
    )
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), k) + 1))
    ndcg = dcg / idcg if idcg else 0.0
    gold_spec_set = {str(value) for value in gold_specs if str(value)}
    top_spec_set = {str(value) for value in list(ranked_specs)[:k] if str(value)}
    gold_spec_acc = len(gold_spec_set & top_spec_set) / len(gold_spec_set) if gold_spec_set else 0.0
    return {
        "acc": hit,
        "gold_clause_coverage": coverage,
        "gold_spec_acc": gold_spec_acc,
        "mrr": reciprocal,
        "ndcg": ndcg,
    }


def coverage_metrics(
    gold_doc_ids: Iterable[int],
    ranked_doc_ids: Iterable[int],
    k: int,
) -> dict[str, float]:
    """Top-10 retrieval-table convention: dedup then compute ACC / Coverage / MRR / nDCG by doc_id."""
    gold = set(int(v) for v in gold_doc_ids)
    ranked: list[int] = []
    for value in list(ranked_doc_ids)[:k]:
        value = int(value)
        if value not in ranked:
            ranked.append(value)
    hit_ranks = [index + 1 for index, doc_id in enumerate(ranked) if doc_id in gold]
    acc = 1.0 if hit_ranks else 0.0
    coverage = len(gold.intersection(ranked)) / len(gold) if gold else 0.0
    mrr = 1.0 / hit_ranks[0] if hit_ranks else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in hit_ranks)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    ndcg = dcg / idcg if idcg else 0.0
    return {"ACC": acc, "Coverage": coverage, "MRR": mrr, "nDCG": ndcg}