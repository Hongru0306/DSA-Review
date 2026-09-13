"""Baselines: BM25 / dense return a full order; adapters exist and map doc-id markers."""

from __future__ import annotations

import pytest

from retrieval import (
    BM25,
    DenseNaive,
    GraphRAGRetriever,
    HippoRAGRetriever,
    LightRAGRetriever,
    RaptorRetriever,
)
from retrieval._support import parse_doc_ids, require, tag_docs

SIMPLE_BASELINES = [BM25, DenseNaive]


@pytest.mark.parametrize("factory", SIMPLE_BASELINES)
def test_simple_baseline_returns_full_order(factory, corpus, encoder):
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
    assert ranking[0] == 0  # the only doc containing cured / fourteen / days


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


def test_simple_baseline_names():
    assert BM25().name == "BM25"
    assert DenseNaive().name == "NaiveRAG"


def test_adapter_classes_expose_names():
    assert GraphRAGRetriever.name == "GraphRAG"
    assert LightRAGRetriever.name == "LightRAG"
    assert HippoRAGRetriever.name == "HippoRAG"
    assert RaptorRetriever.name == "RAPTOR"


def test_docid_tagging_and_parsing():
    tagged = tag_docs(["alpha", "beta"])
    assert tagged[0].startswith("[DOCID:0]")
    assert tagged[1].startswith("[DOCID:1]")
    context = "[DOCID:1] beta text ... [DOCID:0] alpha ... [DOCID:9] out of range"
    assert parse_doc_ids(context, corpus_size=2) == [1, 0]
    assert parse_doc_ids(context, corpus_size=2, k=1) == [1]


def test_require_raises_clear_import_error():
    with pytest.raises(ImportError, match="nonexistent_baseline_module_xyz"):
        require("nonexistent_baseline_module_xyz", "install hint")