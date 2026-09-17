"""Run the retrieval baselines over a corpus, wiring the real-RAG adapters to this
package's LLMClient / SemanticEncoder.

This is the end-to-end entry point: it builds each requested retriever through
``retrieval.factory.build_retriever`` (which bridges the LLM client and encoder
into the framework-specific hooks) and reports retrieval metrics per method.

It needs real endpoints and the optional baseline dependencies:
- LLM:        set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL (OpenAI-compatible).
- Embeddings: EMBEDDING_MODEL (a sentence-transformers / BGE model name; local).
- GraphRAG also needs an HTTP embeddings endpoint: GRAPHRAG_EMBEDDING_API_BASE,
  and GRAPHRAG_EMBEDDING_MODEL (defaults to EMBEDDING_MODEL).

See requirements-baselines.txt for installing the frameworks. BM25 / dense
run without any optional dependency.

Usage:
    python examples/run_baselines.py \
        --corpus data/corpus.jsonl --queries data/queries.jsonl \
        --methods bm25,naive,graphrag,lightrag,hipporag,raptor --top-k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List

# Allow running `python examples/run_baselines.py` from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import mean_100, retrieval_metrics
from retrieval import build_retriever

DEFAULT_METHODS = ["bm25", "naive"]


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_corpus(path: Path) -> List[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [item if isinstance(item, str) else item["text"] for item in data]
    return [row if isinstance(row, str) else row["text"] for row in _load_jsonl(path)]


def _build_encoder() -> Any:
    from encoder import SemanticEncoder

    return SemanticEncoder(model_name=os.environ.get("EMBEDDING_MODEL") or None)


def _build_llm() -> Any:
    from llm import LLMClient

    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        return None
    return LLMClient(
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
    )


def _method_kwargs(method: str, workdir: Path) -> dict:
    if method == "lightrag":
        return {"working_dir": str(workdir / "lightrag")}
    if method == "hipporag":
        return {"save_dir": str(workdir / "hipporag")}
    if method == "graphrag":
        return {
            "workspace": str(workdir / "graphrag"),
            "embedding_api_base": os.environ.get("GRAPHRAG_EMBEDDING_API_BASE")
            or os.environ.get("LLM_BASE_URL"),
            "embedding_model": os.environ.get("GRAPHRAG_EMBEDDING_MODEL")
            or os.environ.get("EMBEDDING_MODEL")
            or "bge-local",
        }
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--workdir", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    corpus = _load_corpus(args.corpus)
    queries = _load_jsonl(args.queries)
    encoder = _build_encoder()
    llm_client = _build_llm()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"corpus_docs={len(corpus)} queries={len(queries)} methods={methods}")
    print(f"{'method':<12}{'Hit@k':>8}{'Recall@k':>10}{'MRR@k':>8}{'nDCG@k':>8}")
    output_rows: List[dict] = []
    for method in methods:
        kwargs = _method_kwargs(method, args.workdir)
        retriever = build_retriever(method, encoder=encoder, llm_client=llm_client, **kwargs)
        retriever.build(corpus, encoder)

        hits, recalls, mrrs, ndcgs = [], [], [], []
        for query in queries:
            ranking = retriever.rank(query["question"])
            metrics = retrieval_metrics(ranking, query.get("gold", []), k=args.top_k)
            hits.append(metrics["Hit@k"])
            recalls.append(metrics["Recall@k"])
            mrrs.append(metrics["MRR@k"])
            ndcgs.append(metrics["nDCG@k"])
            output_rows.append(
                {
                    "method": method,
                    "qid": query.get("qid"),
                    "question": query["question"],
                    "topk": list(ranking[: args.top_k]),
                    **metrics,
                }
            )
        print(
            f"{method:<12}{mean_100(hits):>7.1f}%{mean_100(recalls):>9.1f}%"
            f"{mean_100(mrrs):>7.1f}%{mean_100(ndcgs):>7.1f}%"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(output_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
