"""Prepare the GraphRAG-Bench evaluation subset.

Reads the benchmark from the configured source (a local directory or the
GraphRAG-Bench dataset on HuggingFace), maps it to the pipeline schema, selects
the evaluation sample deterministically, and writes the prepared artifacts:

    <prepared_dir>/corpus.jsonl    {"doc_id", "text", "source", "title"}
    <prepared_dir>/queries.jsonl   {"qid", "question", "answer", "gold": [doc_id], "group"}
    <prepared_dir>/manifest.json   counts, source, sample, versions

Usage:
    python -X utf8 -m benchmarks.graphrag_bench.prepare --config configs/graphrag_bench.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from benchmarks.graphrag_bench import common
from utils.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the GraphRAG-Bench evaluation subset")
    parser.add_argument("--config", type=Path, default=common.DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None, help="keep only the first N selected queries")
    return parser.parse_args()


def _field_map(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "question": "question",
        "answer": "answer",
        "passages": "passages",
        "context": "context",
        "gold": "gold",
        "group": "group",
        "qid": "qid",
        "title": "title",
        "source": "source",
    }
    defaults.update({k: v for k, v in (config.get("data", {}).get("field_map", {}) or {}).items()})
    return defaults


def _passage_text(passage: Any, fields: dict[str, str]) -> str:
    if isinstance(passage, str):
        return passage
    if isinstance(passage, dict):
        for key in (fields["context"], "text", "passage", "content", "paragraph"):
            if passage.get(key):
                return str(passage[key])
    return ""


def _corpus_from_passages(passages: list[Any], fields: dict[str, str]) -> list[dict[str, Any]]:
    """Flatten a benchmark's passages into one row per passage with contiguous doc_id."""
    corpus: list[dict[str, Any]] = []
    for item in passages:
        if isinstance(item, (list, tuple)):
            candidates = list(item)
        else:
            candidates = [item]
        for passage in candidates:
            text = _passage_text(passage, fields).strip()
            if not text:
                continue
            row: dict[str, Any] = {"doc_id": len(corpus), "text": text}
            if isinstance(passage, dict):
                if passage.get(fields["title"]):
                    row["title"] = str(passage[fields["title"]])
                if passage.get(fields["source"]):
                    row["source"] = str(passage[fields["source"]])
            corpus.append(row)
    return corpus


def _resolve_gold(gold: Any, text_to_doc: dict[str, int], corpus_size: int) -> list[int]:
    if gold is None:
        return []
    values = gold if isinstance(gold, (list, tuple)) else [gold]
    resolved: list[int] = []
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            if 0 <= value < corpus_size and value not in resolved:
                resolved.append(value)
        elif isinstance(value, str):
            doc_id = text_to_doc.get(value.strip())
            if doc_id is not None and doc_id not in resolved:
                resolved.append(doc_id)
    return resolved


def load_local(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = common.data_section(config)
    local_dir = Path(data.get("local_dir", "data/graphrag_bench"))
    corpus_path = Path(data.get("corpus_file", local_dir / common.PREPARED_CORPUS))
    queries_path = Path(data.get("queries_file", local_dir / common.PREPARED_QUERIES))
    if not corpus_path.exists() or not queries_path.exists():
        raise SystemExit(
            f"local data not found: {corpus_path} / {queries_path}.\n"
            "Point data.local_dir at a directory containing corpus.jsonl and queries.jsonl, "
            "or set data.source: huggingface."
        )
    corpus = read_jsonl(corpus_path)
    queries = read_jsonl(queries_path)
    for index, row in enumerate(corpus):
        row["doc_id"] = int(row.get("doc_id", index))
        row["text"] = str(row.get("text", ""))
    return corpus, queries


def load_huggingface(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "the `datasets` package is required for data.source: huggingface "
            "(pip install datasets)"
        ) from exc

    data = common.data_section(config)
    fields = _field_map(config)
    dataset = load_dataset(
        data.get("hf_repo", "GraphRAG-Bench/GraphRAG-Bench"),
        data.get("hf_config") or config.get("subset"),
        split=data.get("split", "test"),
    )

    corpus: list[dict[str, Any]] = []
    text_to_doc: dict[str, int] = {}
    queries: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        passages = row.get(fields["passages"], []) or []
        rows = _corpus_from_passages(passages, fields)
        start = len(corpus)
        for item in rows:
            item["doc_id"] = len(corpus)
            corpus.append(item)
        for item in rows:
            text_to_doc.setdefault(item["text"], item["doc_id"])
        gold = _resolve_gold(row.get(fields["gold"]), text_to_doc, len(corpus))
        if not gold and start < len(corpus):
            gold = [start]
        queries.append(
            {
                "qid": str(row.get(fields["qid"], index)),
                "question": str(row.get(fields["question"], "")).strip(),
                "answer": str(row.get(fields["answer"], "")).strip(),
                "gold": gold,
                "group": row.get(fields["group"]),
            }
        )
    return corpus, [q for q in queries if q["question"]]


def select_sample(queries: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministically select the evaluation sample (reproducible across runs)."""
    sample = config.get("data", {}).get("sample", {}) or {}
    size = sample.get("size")
    seed = int(sample.get("seed", 20260101))
    if not size or int(size) <= 0 or int(size) >= len(queries):
        return list(queries)
    rng = random.Random(seed)
    picked = rng.sample(queries, int(size))
    order = {q["qid"]: i for i, q in enumerate(queries)}
    return sorted(picked, key=lambda q: order[q["qid"]])


def main() -> None:
    args = parse_args()
    config = common.load_config(args.config)
    source = str(common.data_section(config).get("source", "local")).lower()
    corpus, queries = load_huggingface(config) if source == "huggingface" else load_local(config)

    before = len(queries)
    queries = select_sample(queries, config)
    if args.limit:
        queries = queries[: args.limit]

    manifest = {
        "name": config.get("name", "graphrag_bench"),
        "subset": config.get("subset"),
        "source": source,
        "corpus_rows": len(corpus),
        "query_rows_before_sample": before,
        "query_rows": len(queries),
        "sample": config.get("data", {}).get("sample", {}),
        "config_path": config.get("_config_path"),
        "versions": common.code_version(),
        "data": common.data_version(corpus, queries),
    }
    common.write_prepared(config, corpus, queries, manifest)
    print(
        f"prepared {len(corpus)} passages / {len(queries)} queries -> {common.prepared_dir(config)}"
    )


if __name__ == "__main__":
    main()
