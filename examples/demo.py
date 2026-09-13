"""全链路演示:合成语料 -> 确定性构图 -> 全部检索方法 + 指标。

无需数据文件、无需下载模型、无需 LLM。运行:
    python examples/demo.py
(Windows 控制台若中文乱码,先 ``set PYTHONIOENCODING=utf-8`` 或只观察指标数字)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# 保证从任意目录直接运行 ``python examples/demo.py`` 时能 import 顶层模块
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation import retrieval_metrics
from graph import build_corpus_graph
from retrieval import (
    BM25,
    DenseNaive,
    GraphRAGLite,
    HippoRAGLite,
    LightRAGLite,
    OursRetriever,
    RAPTORLite,
)


class DummyEncoder:
    """确定性假编码器(按文本 SHA-1 种子生成 L2 归一化向量),仅供演示。"""

    dim = 64

    def encode_batch(self, texts):
        out = []
        for text in texts:
            digest = hashlib.sha1(str(text).encode("utf-8")).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            vector = rng.normal(size=self.dim).astype(np.float32)
            norm = np.linalg.norm(vector)
            out.append(vector / norm if norm > 1e-9 else vector)
        return np.stack(out)


CORPUS = [
    "混凝土构件浇筑完成后应进行保温保湿养护，养护时间不应少于14天。",
    "钢筋进场时应核验出厂合格证与复试报告，见证取样比例如表规定。",
    "大体积混凝土施工应控制浇筑温度，其内外温差不宜超过25℃。",
    "模板拆除应符合混凝土强度达到设计要求后方可进行。",
    "脚手架搭设应满足稳定性要求，立杆间距与步距按方案执行。",
]
QUESTIONS = [
    ("q1", "养护时间不应少于几天？", [0]),
    ("q2", "内外温差不宜超过多少？", [2]),
    ("q3", "模板支撑何时可以拆除？", [3]),
]


def main() -> None:
    encoder = DummyEncoder()
    corpus_graph = build_corpus_graph(CORPUS)

    ours = OursRetriever(n=3, alpha=0.03, lambda_=0.85)
    ours.build(CORPUS, encoder, corpus_graph=corpus_graph)

    methods = {
        "Ours": ours,
        "BM25": BM25(),
        "NaiveRAG": DenseNaive(),
        "GraphRAG-lite": GraphRAGLite(),
        "HippoRAG-lite": HippoRAGLite(),
        "LightRAG-lite": LightRAGLite(),
        "RAPTOR-lite": RAPTORLite(),
    }
    for method in methods.values():
        method.build(CORPUS, encoder)

    print(f"corpus_docs={len(CORPUS)} questions={len(QUESTIONS)} methods={len(methods)}")
    print(f"{'method':<14}{'Hit@3':>8}{'MRR@3':>9}{'nDCG@3':>9}{'top3_for_q1':>18}")
    for name, method in methods.items():
        hit_tot = mrr_tot = ndcg_tot = 0.0
        top1_q1 = None
        for qid, question, gold in QUESTIONS:
            ranking = method.rank(question)
            metrics = retrieval_metrics(ranking, gold, k=3)
            hit_tot += metrics["Hit@k"]
            mrr_tot += metrics["MRR@k"]
            ndcg_tot += metrics["nDCG@k"]
            if qid == "q1":
                top1_q1 = ranking[0]
        n = len(QUESTIONS)
        print(f"{name:<14}{hit_tot / n * 100:>7.1f}%{mrr_tot / n:>9.3f}{ndcg_tot / n:>9.3f}{top1_q1:>18}")


if __name__ == "__main__":
    main()