"""基线检索:均实现 build/rank 协议,返回全序;BM25 内容级判定。"""

from __future__ import annotations

import pytest

from retrieval import (
    BM25,
    DenseNaive,
    GraphRAGLite,
    HippoRAGLite,
    LightRAGLite,
    RAPTORLite,
)

ALL_BASELINES = [BM25, DenseNaive, GraphRAGLite, HippoRAGLite, LightRAGLite, RAPTORLite]


@pytest.mark.parametrize("factory", ALL_BASELINES)
def test_baseline_returns_full_order(factory, corpus, encoder):
    method = factory()
    method.build(corpus, encoder)
    ranking = method.rank("混凝土养护时间不少于14天")
    assert sorted(ranking) == list(range(len(corpus)))


def test_bm25_ascii_corpus_ranks_matching_doc_first(encoder):
    corpus = [
        "structural concrete shall be cured for fourteen days",
        "steel rebar delivery requires inspection records",
        "concrete temperature gap shall not exceed twenty five degree",
        "formwork removal after concrete reaches design strength",
    ]
    bm25 = BM25()
    bm25.build(corpus, encoder)
    ranking = bm25.rank("concrete cured fourteen days")
    assert ranking[0] == 0  # 唯一同时含 cured / fourteen / days 的文档


def test_bm25_no_overlap_terms_still_full_rank(corpus, encoder):
    bm25 = BM25()
    bm25.build(corpus, encoder)
    ranking = bm25.rank("完全无关词汇xyz")
    assert sorted(ranking) == list(range(len(corpus)))


def test_dense_topk_is_hit_or_trivial(corpus, encoder):
    dense = DenseNaive()
    dense.build(corpus, encoder)
    top3 = dense.rank("大体积混凝土温差不应超过25℃")[:3]
    assert len(top3) == 3
    assert len(set(top3)) == 3


def test_method_names_present():
    expected = {
        "BM25": "BM25",
        "DenseNaive": "NaiveRAG",
        "GraphRAGLite": "GraphRAG-lite",
        "HippoRAGLite": "HippoRAG-lite",
        "LightRAGLite": "LightRAG-lite",
        "RAPTORLite": "RAPTOR-lite",
    }
    for factory, name in expected.items():
        assert getattr(__import__("retrieval", fromlist=[factory]), factory)().name == name