"""Hypergraph-RAG core algorithm library.

Modular core package: offline graph construction -> retrieval (Ours subgraph
retrieval + baselines) -> generation (evidence QA / clause-grounded review) ->
evaluation (retrieval / generation / revision metrics).
No data files are included; all inputs are passed as JSON dicts / ``list[str]``.
"""

from config import (
    DEFAULT_FROZEN_REPUBLISH_PARAMS,
    DEFAULT_PAPER_PARAMS,
    MAX_DOC_ENT,
    MAX_DOC_REL,
    MAX_Q,
    MAX_REL,
    STOP_EN,
    STOP_ZH,
)
from encoder import SemanticEncoder
from retrieval.bm25 import BM25
from retrieval.dense import DenseNaive
from retrieval.graphrag import GraphRAGRetriever
from retrieval.hipporag import HippoRAGRetriever
from retrieval.lightrag import LightRAGRetriever
from retrieval.ours import OursRetriever
from retrieval.ours_torch import OursRetrieverTorch
from retrieval.raptor import RaptorRetriever

__all__ = [
    "DEFAULT_FROZEN_REPUBLISH_PARAMS",
    "DEFAULT_PAPER_PARAMS",
    "MAX_DOC_ENT",
    "MAX_DOC_REL",
    "MAX_Q",
    "MAX_REL",
    "STOP_EN",
    "STOP_ZH",
    "SemanticEncoder",
    "OursRetriever",
    "OursRetrieverTorch",
    "DenseNaive",
    "BM25",
    "GraphRAGRetriever",
    "LightRAGRetriever",
    "HippoRAGRetriever",
    "RaptorRetriever",
]