"""pytest shared fixtures: deterministic fake encoder + tiny synthetic corpus, no downloads or network."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

# Tiny Chinese clause corpus (4 docs)
CORPUS = [
    "混凝土构件浇筑完成后应进行保温保湿养护，养护时间不应少于14天。",
    "钢筋进场时应核验出厂合格证与复试报告，见证取样比例如表规定。",
    "大体积混凝土施工应控制浇筑温度，其内外温差不宜超过25℃。",
    "模板拆除应符合混凝土强度达到设计要求后方可进行。",
]


class FakeEncoder:
    """Deterministic L2-normalized embeddings seeded by text SHA-1 (inner product = cosine)."""

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
    """Fake LLM returning a preset response, for client tests."""

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