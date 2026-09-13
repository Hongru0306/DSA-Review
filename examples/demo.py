"""End-to-end demo: synthetic corpus -> deterministic graph -> retrievers + metrics.

No data files, no model downloads, no LLM required. Run:
    python core/examples/demo.py
(If Chinese output is garbled on Windows, `set PYTHONIOENCODING=utf-8` first.)

The real-RAG baseline adapters (GraphRAG / LightRAG / HippoRAG / RAPTOR) are not
run here because they need their optional dependencies and an LLM endpoint; the
offline Ours / BM25 / dense methods run without them. See requirements-baselines.txt.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Allow running `python core/examples/demo.py` from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation import retrieval_metrics
from graph import build_corpus_graph
from retrieval import BM25, DenseNaive, OursRetriever


class DummyEncoder:
    """Deterministic fake encoder (SHA-1-seeded L2-normalized vectors), demo only."""

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