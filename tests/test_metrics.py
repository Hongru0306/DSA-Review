"""Evaluation metrics: hand-verifiable ground-truth assertions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation import (
    bleu1,
    char_prf,
    char_scores,
    coverage_metrics,
    exact_match,
    gen_avg,
    macro_mean,
    mean_100,
    ret_avg,
    retrieval_metrics,
    review_retrieval_metrics,
    rouge_l_f1,
    token_prf,
    token_scores,
)
from evaluation.revision import (
    delta_embedding_prf,
    embedding_prf,
    revision_diff_char_prf,
    revision_edit_fragments,
)


def test_retrieval_metrics_ground_truth():
    metrics = retrieval_metrics(ranked=[1, 0, 2, 3], gold=[2], k=5)
    assert metrics["MRR@k"] == pytest.approx(1 / 3)
    assert metrics["Hit@k"] == 1.0
    assert metrics["Recall@k"] == 1.0
    assert metrics["Precision@k"] == pytest.approx(1 / 5)
    assert metrics["nDCG@k"] == pytest.approx(1 / math.log2(4))
    assert metrics["MAP@k"] == pytest.approx(1 / 3)


def test_review_retrieval_metrics_ground_truth():
    metrics = review_retrieval_metrics([2], [1, 2, 3], 3)
    assert metrics["acc"] == 1.0
    assert metrics["gold_clause_coverage"] == 1.0
    assert metrics["mrr"] == pytest.approx(0.5)
    dcg = 1 / math.log2(3)
    assert metrics["ndcg"] == pytest.approx(dcg)


def test_review_retrieval_metrics_spec_acc():
    metrics = review_retrieval_metrics(
        [2], [1, 2, 3], 3,
        gold_specs=["GB50010"], ranked_specs=["GB50010", "B", "C"],
    )
    assert metrics["gold_spec_acc"] == 1.0


def test_coverage_metrics_dedup():
    metrics = coverage_metrics(gold_doc_ids=[2, 5], ranked_doc_ids=[1, 2, 2, 5], k=4)
    assert metrics["ACC"] == 1.0
    assert metrics["Coverage"] == pytest.approx(1.0)
    assert metrics["MRR"] == pytest.approx(0.5)


def test_char_prf_ground_truth():
    p, r, f = char_prf("混凝土", "混凝土构件")
    assert (p, r) == (1.0, pytest.approx(0.6))
    assert f == pytest.approx(2 * 1.0 * 0.6 / 1.6)


def test_token_prf_squd_style():
    p, r, f = token_prf("The concrete strength", "concrete strength")
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(1.0)
    assert f == pytest.approx(1.0)


def test_char_and_token_scores():
    assert char_scores("混凝土", "混凝土构件")[2] == pytest.approx(0.75)
    assert token_scores("A B C", "A B C")[2] == pytest.approx(1.0)


def test_rouge_l_and_bleu():
    assert rouge_l_f1("A B C", "A B D") == pytest.approx(2 / 3)
    assert bleu1("A B C", "A B C") == pytest.approx(1.0)
    assert bleu1("A B C", "A B D") == pytest.approx(2 / 3)


def test_exact_match():
    assert exact_match("混凝土", "混凝土 ") is True
    assert exact_match("混凝土", "构件") is False


def test_revision_edit_fragments_and_diff():
    fragments = revision_edit_fragments("甲乙丙", "甲乙丁")
    assert fragments == ["丁"]
    p, r, f, pred_frags, ref_frags = revision_diff_char_prf("甲乙丙", "甲乙丁", "甲乙丁")
    assert f == pytest.approx(1.0)
    assert pred_frags == ["丁"]


def test_embedding_prf_identical(encoder):
    emb = {t: encoder.encode_single(t) for t in ["甲乙丙", "甲乙丁"]}
    p, r, f = embedding_prf(["甲乙丁"], ["甲乙丁"], emb)
    assert p == pytest.approx(1.0)
    assert f == pytest.approx(1.0)


def test_delta_embedding_prf_same_edit(encoder):
    emb = {t: encoder.encode_single(t) for t in ["甲乙丙", "甲乙丁"]}
    p, r, f = delta_embedding_prf("甲乙丙", "甲乙丁", "甲乙丁", emb)
    assert p == pytest.approx(1.0, abs=1e-6)
    assert f == pytest.approx(1.0, abs=1e-6)


def test_aggregate_helpers():
    assert macro_mean([1.0, 2.0, 3.0]) == 2.0
    assert mean_100([0.25, 0.75]) == 50.0
    assert ret_avg(0.8, 0.5, 0.7) == pytest.approx((0.8 + 0.5 + 0.7) / 3)
    assert gen_avg([0.5, 0.5, 1.0]) == pytest.approx(2 / 3)