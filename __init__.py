"""core —— 超图 RAG 核心算法库。

模块化整理后的核心算法包:离线构图 -> 检索(Ours 子图检索 + 对比基线) ->
生成(证据问答 / 条款归因审查) -> 评测(检索 / 生成 / 修订指标)。
不包含任何数据文件;所有输入以 JSON dict / list[str] 形式传入。
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
from retrieval.lightrag_lite import LightRAGLite
from retrieval.ours import OursRetriever
from retrieval.ours_torch import OursRetrieverTorch
from retrieval.ppr import GraphRAGLite, HippoRAGLite
from retrieval.raptor import RAPTORLite

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
    "GraphRAGLite",
    "HippoRAGLite",
    "LightRAGLite",
    "RAPTORLite",
]