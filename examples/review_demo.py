"""Document review example: retrieve supporting clauses for a passage and produce a review result.

The passage and its surrounding context are turned into a retrieval query; the
retrieved clauses are formatted into an evidence block and passed to the LLM,
which returns an assessment (compliant / has error) plus a suggested revision.
When the example case carries a reference revision, the result is also scored.

Models must be configured first (see README): ``LLM_API_KEY`` / ``LLM_BASE_URL`` /
``LLM_MODEL`` and ``EMBEDDING_MODEL``. With ``--no-generate`` the script only
retrieves and skips the LLM step, so it can run without an LLM endpoint.

Usage:
    python -X utf8 examples/review_demo.py \
        --input examples/review_example.json \
        --output outputs/example/review.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Allow running `python examples/review_demo.py` from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder import SemanticEncoder
from generation.review import (
    dual_generation_score_fields,
    extract_predicted_correct_sentence,
    generate_review,
    parse_has_error,
    review_generation_prompt,
)
from graph import build_corpus_graph
from retrieval import build_retriever
from utils.io import read_json, write_json

NOT_CONFIGURED = (
    "LLM_API_KEY is not set. Configure the models first (see README), "
    "or run with --no-generate to retrieve evidence only."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DSA-Review document review example")
    parser.add_argument("--input", type=Path, default=Path("examples/review_example.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/example/review.json"))
    parser.add_argument("--method", default="ours", help="retriever name (ours, bm25, naive, graphrag, ...)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--workdir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--corpus-graph",
        type=Path,
        default=None,
        help="prebuilt corpus graph JSON; built deterministically when omitted",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="retrieve evidence only; skip the LLM review step",
    )
    return parser.parse_args()


def build_llm_client() -> Any:
    from llm import LLMClient

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(NOT_CONFIGURED)
    return LLMClient(
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
    )


def retriever_kwargs(method: str, workdir: Path) -> dict:
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
    args = parse_args()
    data = read_json(args.input)
    corpus_rows = data["corpus"]
    corpus = [row["text"] for row in corpus_rows]
    case = dict(data["case"])

    encoder = SemanticEncoder(model_name=os.environ.get("EMBEDDING_MODEL") or None)
    llm_client = None if args.no_generate else build_llm_client()
    corpus_graph = read_json(args.corpus_graph) if args.corpus_graph else build_corpus_graph(corpus)

    retriever = build_retriever(
        args.method, encoder=encoder, llm_client=llm_client, **retriever_kwargs(args.method, args.workdir)
    )
    if args.method == "ours":
        retriever.build(corpus, encoder, corpus_graph=corpus_graph)
    else:
        retriever.build(corpus, encoder)

    ranking = retriever.rank(case["review_sentence"])[: args.top_k]
    evidence = []
    for rank, index in enumerate(ranking, 1):
        row = dict(corpus_rows[index])
        row.setdefault("doc_id", index)
        row["rank"] = rank
        evidence.append(row)
    retrieval_row = {"method": retriever.name, "retrieved_top10": evidence}

    result: dict[str, Any] = {
        "method": retriever.name,
        "query": case["review_sentence"],
        "scenario_context": case.get("scenario_context", ""),
        "retrieval": {"top_k": args.top_k, "evidence": evidence},
    }
    print(f"method={retriever.name} retrieved={len(evidence)}")
    for row in evidence:
        print(f"  [{row['rank']}] {row.get('spec_name', '')} {row.get('clause', '')}")

    if not args.no_generate:
        prompt = review_generation_prompt(case, retrieval_row)
        prediction = generate_review(llm_client, case, retrieval_row)
        revision = extract_predicted_correct_sentence(prediction, case["review_sentence"])
        result["assessment"] = {
            "has_error": parse_has_error(prediction),
            "suggested_revision": revision,
            "raw_response": prediction,
        }
        result["prompt_chars"] = len(prompt)
        print(f"assessment.has_error={result['assessment']['has_error']}")
        print(f"suggested_revision={revision}")

        if case.get("correct_sentence"):
            fields = dual_generation_score_fields(prediction, case)
            result["metrics"] = {
                "revision_char_f1": fields["revision_char_f1"],
                "revision_diff_char_f1": fields["revision_diff_char_f1"],
                "full_answer_char_f1": fields["full_answer_char_f1"],
            }
            print(f"revision_char_f1={fields['revision_char_f1']:.4f}")

    write_json(args.output, result)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
