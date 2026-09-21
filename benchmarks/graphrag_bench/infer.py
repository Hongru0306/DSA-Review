"""Run retrieval (and optional answer generation) over the prepared benchmark.

For every configured method and query this writes one prediction row: the ranked
evidence plus, when generation is enabled and an LLM endpoint is configured, the
generated answer. Retrieval and evaluation are separate steps, so saved
predictions can be rescored without repeating generation.

Usage:
    python -X utf8 -m benchmarks.graphrag_bench.infer \
        --config configs/graphrag_bench.yaml --output-dir outputs/graphrag_bench
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from benchmarks.graphrag_bench import common
from generation.qa import generate_answer
from utils.io import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on the prepared benchmark")
    parser.add_argument("--config", type=Path, default=common.DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--methods", default=None, help="comma-separated override of retrieval.methods")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N queries")
    parser.add_argument("--no-generate", action="store_true", help="retrieval only; skip answer generation")
    return parser.parse_args()


def _methods(config: dict[str, Any], override: str | None) -> list[str]:
    if override:
        return [m.strip().lower() for m in override.split(",") if m.strip()]
    return common.method_names(config)


def main() -> None:
    args = parse_args()
    config = common.load_config(args.config)
    common.require_prepared(config)

    corpus = common.load_corpus(config)
    queries = common.load_queries(config)
    if args.limit:
        queries = queries[: args.limit]
    texts = [row["text"] for row in corpus]
    doc_by_id = {int(row["doc_id"]): row for row in corpus}

    encoder = common.build_encoder(config)
    gen_cfg = common.generation_section(config)
    generate = bool(gen_cfg.get("enabled", False)) and not args.no_generate
    llm_client = common.build_llm(config, required=generate)
    corpus_graph = common.build_graph(config, texts)

    methods = _methods(config, args.methods)
    top_k = common.top_k(config)
    gen_top_k = int(gen_cfg.get("top_k", top_k))
    output_dir = Path(args.output_dir) if args.output_dir else common.default_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for method in methods:
        retriever = common.build_method(method, config, encoder, llm_client, texts, corpus_graph)
        label = getattr(retriever, "name", method)
        failures = 0
        for query in queries:
            ranking = [int(i) for i in retriever.rank(query["question"])[:top_k]]
            evidence = [dict(doc_by_id[i], rank=r) for r, i in enumerate(ranking, 1) if i in doc_by_id]
            row: dict[str, Any] = {
                "method": method,
                "resolved_method": label,
                "qid": query.get("qid"),
                "question": query["question"],
                "answer": query.get("answer", ""),
                "gold": query.get("gold", []),
                "group": query.get("group"),
                "topk": ranking,
                "retrieved_top10": evidence,
            }
            if generate and llm_client is not None:
                qa_row = {
                    "qid": query.get("qid"),
                    "question": query["question"],
                    "gold": query.get("gold", []),
                    "retrieved_top10": evidence[:gen_top_k],
                }
                try:
                    out = generate_answer(llm_client, qa_row, doc_by_id, label, top_k=gen_top_k)
                    row["prediction"] = out["long_answer"]
                except Exception as exc:  # keep the run going; record the failure
                    failures += 1
                    row["prediction"] = ""
                    row["generation_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        print(f"{method}: {len(queries)} queries" + (f", {failures} generation failures" if failures else ""))

    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_json(
        output_dir / "run_config.json",
        common.run_config_snapshot(
            config,
            {
                "methods": methods,
                "predictions": len(rows),
                "queries": len(queries),
                "generation_enabled": bool(generate and llm_client is not None),
                "data": common.data_version(corpus, queries),
            },
        ),
    )
    print(f"wrote {output_dir / 'predictions.jsonl'} ({len(rows)} rows)")
    print(f"wrote {output_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
