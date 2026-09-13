"""pytest 共享夹具:确定性假编码器 + 微型合成语料/构图,免下载免联网。"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from graph.builder import build_corpus_graph

# 微型中文条款语料(4 篇,覆盖共现实体与关系)
CORPUS = [
    "混凝土构件浇筑完成后应进行保温保湿养护，养护时间不应少于14天。",
    "钢筋进场时应核验出厂合格证与复试报告，见证取样比例如表规定。",
    "大体积混凝土施工应控制浇筑温度，其内外温差不宜超过25℃。",
    "模板拆除应符合混凝土强度达到设计要求后方可进行。",
]


class FakeEncoder:
    """按文本 SHA-1 种子生成确定性 L2 归一化嵌入(内积即余弦)。"""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        digest = hashlib.sha1(str(text).encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        vector = rng.normal(size=self.dim).astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-9 else vector

    def encode_batch(self, texts) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def encode_single(self, text: str) -> np.ndarray:
        return self._vec(text)


class FakeLLM:
    """返回预置响应的假 LLM,客户端测试用。"""

    def __init__(self, text: str = "根据“混凝土”明确养护时间不少于14天。"):
        self.text = text
        self.calls = []

    def call_text(self, system: str, user: str, **kwargs) -> str:
        self.calls.append((system, user))
        return self.text


@pytest.fixture
def encoder():
    return FakeEncoder()


@pytest.fixture
def corpus():
    return list(CORPUS)


@pytest.fixture
def corpus_graph(corpus):
    return build_corpus_graph(corpus)