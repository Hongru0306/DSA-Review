"""Ours subgraph retrieval: hand-derived scoring, full ordering, parameter defaults."""

from __future__ import annotations

import numpy as np
import pytest

from config import DEFAULT_PAPER_PARAMS
from retrieval.ours import OursRetriever
from retrieval.ours_torch import OursRetrieverTorch
from utils.text import doc_cache_key


def _tiny_setup(encoder):
    """Two single-entity docs plus explicit corpus_graph and qcache, so everything is hand-derivable."""
    corpus = ["AAA", "BBB"]
    corpus_graph = {
        doc_cache_key("AAA"): {"entities": ["AAA"], "relations": []},
        doc_cache_key("BBB"): {"entities": ["BBB"], "relations": []},
    }
    qcache = {"AAA是什么": {"entities": ["AAA"], "relations": []}}
    retriever = OursRetriever(
        n=3, m=2, gamma=0.7, alpha=0.03, lambda_=0.85,
        use_llm=True, use_rel=False,
    )
    retriever.build(corpus, encoder, corpus_graph=corpus_graph, qcache=qcache)
    return retriever, corpus


def _manual_scores(retriever, encoder):
    """Independently recompute the paper score (two single-entity docs)."""
    enc_a = encoder.encode_single("AAA")
    enc_b = encoder.encode_single("BBB")
    c = float(np.dot(enc_a, enc_b))
    idfn = float(np.log(1 + 2 / max(1.0, 0.5))) ** 0.25  # soft_idf=log(1+2/1)
    w_A = idfn * (1 + c) / 2
    alpha = 0.03
    scores = np.array(
        [max(idfn - alpha, 0.0) / max(w_A, 1e-9), max(c * idfn - alpha, 0.0) / max(w_A, 1e-9)]
    )
    return scores


def test_manual_score_matches_formula(encoder):
    retriever, _ = _tiny_setup(encoder)
    scores = retriever.score("AAA是什么")
    expected = _manual_scores(retriever, encoder)
    assert scores.shape == (2,)
    assert np.allclose(scores, expected, rtol=1e-5)


def test_rank_is_full_order_and_matches_scores(encoder):
    retriever, corpus = _tiny_setup(encoder)
    ranking = retriever.rank("AAA是什么")
    assert sorted(ranking) == list(range(len(corpus)))
    assert all(type(index) is int for index in ranking)
    expected = _manual_scores(retriever, encoder)
    assert ranking == list(np.argsort(expected)[::-1])


def test_default_params_match_config(encoder, corpus):
    retriever = OursRetriever.from_params(DEFAULT_PAPER_PARAMS)
    assert retriever.n == 3
    assert retriever.alpha == 0.03
    assert retriever.lambda_ == 0.85
    retriever.build(corpus, encoder)
    scores = retriever.score("养护时间不少于14天")
    assert scores.shape == (len(corpus),)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)


def test_paper_and_default_formulas_finite_and_rank(encoder, corpus, corpus_graph):
    retriever = OursRetriever(n=3, use_rel=True)
    retriever.build(corpus, encoder, corpus_graph=corpus_graph)
    ranking = retriever.rank("养护时间不少于14天")
    assert sorted(ranking) == list(range(len(corpus)))
    assert len(set(retriever.retrieve("养护时间不少于14天", top_k=3))) == 3


def test_use_semantic_ablation_binary(encoder, corpus, corpus_graph):
    retriever = OursRetriever(n=3, use_semantic=False)
    retriever.build(corpus, encoder, corpus_graph=corpus_graph)
    scores = retriever.score("养护时间")
    assert np.all(np.isfinite(scores))


def test_ours_torch_agrees_with_numpy(encoder, corpus, corpus_graph):
    torch = pytest.importorskip("torch")
    numpy_retriever = OursRetriever(n=3, use_rel=True)
    numpy_retriever.build(corpus, encoder, corpus_graph=corpus_graph)
    numpy_scores = numpy_retriever.score("养护时间不少于14天")

    torch_retriever = OursRetrieverTorch(n=3, use_rel=True)
    torch_retriever.build(corpus, encoder, corpus_graph=corpus_graph)
    torch_retriever.to_torch("cpu")
    torch_scores = torch_retriever.score_torch("养护时间不少于14天").cpu().numpy()
    assert np.allclose(numpy_scores, torch_scores, rtol=1e-5)