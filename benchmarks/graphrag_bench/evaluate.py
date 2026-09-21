"""Score saved predictions and aggregate the benchmark results.

Retrieval metrics are computed at every ``k`` in ``evaluation.k``; generation
metrics are added for rows that carry a prediction. Per-query metrics are
macro-averaged (x100) per method, overall and per configured grouping field.

Usage:
    python -X utf8 -m benchmarks.graphrag_bench.evaluate --config configs/graphrag_bench.yaml \
        --predictions outputs/graphrag_bench/predictions.jsonl --output outputs/graphrag_bench/metrics.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.graphrag_bench import common
from evaluation import char_prf, exact_match, mean_100, ret_avg, retrieval_metrics, token_prf
from utils.io import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved benchmark predictions")
    parser.add_argument("--config", type=Path, default=common.DEFAULT_CONFIG)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def retrieval_record(row: dict[str, Any], ks: list[int]) -> dict[str, float]:
    record: dict[str, float] = {}
    ranked = row.get("topk") or []
    gold = row.get("gold") or []
    for k in ks:
        metrics = retrieval_metrics(ranked, gold, k=k)
        hit = metrics["Hit@k"]
        record[f"acc@{k}"] = hit
        record[f"coverage@{k}"] = metrics["Recall@k"]
        record[f"precision@{k}"] = metrics["Precision@k"]
        record[f"recall@{k}"] = metrics["Recall@k"]
        record[f"mrr@{k}"] = metrics["MRR@k"]
        record[f"ndcg@{k}"] = metrics["nDCG@k"]
        record[f"map@{k}"] = metrics["MAP@k"]
        record[f"ret_avg@{k}"] = ret_avg(hit, metrics["MRR@k"], metrics["nDCG@k"])
    return record


def generation_record(row: dict[str, Any]) -> dict[str, float]:
    prediction = row.get("prediction")
    answer = row.get("answer") or ""
    if not prediction or not answer:
        return {}
    char_precision, char_recall, char_f1 = char_prf(prediction, answer)
    token_precision, token_recall, token_f1 = token_prf(prediction, answer)
    return {
        "gen_char_precision": char_precision,
        "gen_char_recall": char_recall,
        "gen_char_f1": char_f1,
        "gen_token_precision": token_precision,
        "gen_token_recall": token_recall,
        "gen_token_f1": token_f1,
        "gen_em": 1.0 if exact_match(prediction, answer) else 0.0,
    }


def aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for record in records for key in record})
    return {key: mean_100([r[key] for r in records if key in r]) for key in keys}


def main() -> None:
    args = parse_args()
    config = common.load_config(args.config)
    evaluation = common.evaluation_section(config)
    ks = [int(k) for k in (evaluation.get("k") or [5])]
    group_by = [str(field) for field in (evaluation.get("group_by") or [])]

    rows = read_jsonl(args.predictions)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get("method"))].append(row)

    methods_payload: dict[str, Any] = {}
    for method, items in sorted(by_method.items()):
        per_query: list[dict[str, float]] = []
        for row in items:
            record = retrieval_record(row, ks)
            record.update(generation_record(row))
            per_query.append(record)
        payload: dict[str, Any] = {
            "num_queries": len(items),
            "num_with_prediction": sum(1 for row in items if row.get("prediction")),
            "overall": aggregate(per_query),
        }
        for field in group_by:
            groups: dict[str, list[dict[str, float]]] = defaultdict(list)
            for row, record in zip(items, per_query):
                groups[str(row.get(field))].append(record)
            payload[f"by_{field}"] = {
                group: {"num_queries": len(records), **aggregate(records)}
                for group, records in sorted(groups.items())
            }
        methods_payload[method] = payload

    result = {
        "methods": methods_payload,
        "meta": {
            "config_path": config.get("_config_path"),
            "predictions": str(args.predictions),
            "k": ks,
            "group_by": group_by,
            "versions": common.code_version(),
        },
    }
    write_json(args.output, result)

    primary_k = ks[0]
    print(f"{'method':<10}{'queries':>8}{f'acc@{primary_k}':>9}{f'mrr@{primary_k}':>9}"
          f"{f'ndcg@{primary_k}':>10}{f'retavg@{primary_k}':>11}{'char_f1':>9}{'token_f1':>9}")
    for method, payload in methods_payload.items():
        overall = payload["overall"]
        print(
            f"{method:<10}{payload['num_queries']:>8}"
            f"{overall.get(f'acc@{primary_k}', 0.0):>9.2f}"
            f"{overall.get(f'mrr@{primary_k}', 0.0):>9.2f}"
            f"{overall.get(f'ndcg@{primary_k}', 0.0):>10.2f}"
            f"{overall.get(f'ret_avg@{primary_k}', 0.0):>11.2f}"
            f"{overall.get('gen_char_f1', 0.0):>9.2f}"
            f"{overall.get('gen_token_f1', 0.0):>9.2f}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
