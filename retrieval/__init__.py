"""Retrieval methods: real-RAG baselines (GraphRAG / LightRAG / HippoRAG / RAPTOR) + BM25 / dense BGE."""

from retrieval.base import Retriever
from retrieval.bm25 import BM25
from retrieval.dense import DenseNaive
from retrieval.factory import RETRIEVER_NAMES, build_retriever
from retrieval.graphrag import GraphRAGRetriever
from retrieval.hipporag import HippoRAGRetriever
from retrieval.lightrag import LightRAGRetriever
from retrieval.raptor import RaptorRetriever

__all__ = [
    "Retriever",
    "DenseNaive",
    "BM25",
    "GraphRAGRetriever",
    "LightRAGRetriever",
    "HippoRAGRetriever",
    "RaptorRetriever",
    "build_retriever",
    "RETRIEVER_NAMES",
]
