"""检索引擎统一协议(实验代码的 duck-typing 协议,显式化为接口)。

检索行 schema:``{"method", "qid"/"question", "topK": [doc_idx, ...],
"metrics": {...}}``。``rank(q)`` 返回全语料排序 ``list[int]``(best-first),
调用方按需切片 ``[:top_k]``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class Retriever(ABC):
    """所有检索器实现的接口。"""

    name = "retriever"

    @abstractmethod
    def build(self, corpus: List[str], encoder) -> None:
        """建立索引(corpus 为条款 / 语料切片文本列表,encoder 提供嵌入)。"""

    @abstractmethod
    def rank(self, q: str) -> List[int]:
        """返回全语料 doc 下标,按相关性降序。"""