"""Benchmark protocol: preparation schema, deterministic sampling, metric records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.graphrag_bench import common, evaluate, prepare

MINIMAL_CONFIG = {
    "retrieval": {"top_k": 3, "methods": ["ours", "bm25"], "ours": {"n": 3}},
    "evaluation": {"k": [5], "group_by": ["group"]},
    "generation": {"enabled": False},
    "models": {"llm": {}, "embedding": {"model_env": "EMBEDDING_MODEL"}},
}


def _write_fixture(root: Path) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus = [{"doc_id": i, "text": t} for i, t in enumerate(["alpha clause", "beta clause", "gamma clause"])]
    queries = [
        {"qid": "q1", "question": "alpha?", "answer": "alpha clause", "gold": [0], "group": "1hop"},
        {"qid": "q2", "question": "beta?", "answer": "beta clause", "gold": [1], "group": "1hop"},
    ]
    for name, rows in (("corpus.jsonl", corpus), ("queries.jsonl", queries)):
        (data_dir / name).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
        )
    return data_dir


def test_load_local_normalizes_doc_ids(tmp_path):
    data_dir = _write_fixture(tmp_path)
    config = {"data": {"source": "local", "local_dir": str(data_dir)}}
    corpus, queries = prepare.load_local(config)
    assert [row["doc_id"] for row in corpus] == [0, 1, 2]
    assert queries[0]["qid"] == "q1"


def test_load_local_missing_data_errors(tmp_path):
    config = {"data": {"source": "local", "local_dir": str(tmp_path / "nope")}}
    with pytest.raises(SystemExit, match="local data not found"):
        prepare.load_local(config)


def test_select_sample_is_deterministic_and_seeded():
    queries = [{"qid": f"q{i}"} for i in range(10)]
    config = {"data": {"sample": {"size": 4, "seed": 7}}}
    first = prepare.select_sample(queries, config)
    second = prepare.select_sample(queries, config)
    assert [q["qid"] for q in first] == [q["qid"] for q in second]
    assert len(first) == 4
    # a different seed selects a different subset
    other = prepare.select_sample(queries, {"data": {"sample": {"size": 4, "seed": 8}}})
    assert [q["qid"] for q in other] != [q["qid"] for q in first]


def test_select_sample_returns_all_when_unset():
    queries = [{"qid": f"q{i}"} for i in range(3)]
    assert prepare.select_sample(queries, {}) == queries
    assert prepare.select_sample(queries, {"data": {"sample": {"size": None}}}) == queries
    assert prepare.select_sample(queries, {"data": {"sample": {"size": 99}}}) == queries


def test_resolve_gold_accepts_indices_and_texts():
    text_to_doc = {"alpha": 0, "beta": 1}
    assert prepare._resolve_gold([1, 0], text_to_doc, 2) == [1, 0]
    assert prepare._resolve_gold(["beta"], text_to_doc, 2) == [1]
    assert prepare._resolve_gold([9, "missing"], text_to_doc, 2) == []
    assert prepare._resolve_gold(None, text_to_doc, 2) == []


def test_retrieval_record_ground_truth():
    record = evaluate.retrieval_record({"topk": [1, 0, 2], "gold": [2]}, [3])
    assert record["acc@3"] == 1.0
    assert record["coverage@3"] == pytest.approx(1.0)
    assert record["precision@3"] == pytest.approx(1 / 3)
    assert record["mrr@3"] == pytest.approx(1 / 3)
    assert record["ndcg@3"] == pytest.approx(0.5)
    assert record["map@3"] == pytest.approx(1 / 3)
    assert record["ret_avg@3"] == pytest.approx((1.0 + 1 / 3 + 0.5) / 3)


def test_generation_record_ground_truth():
    record = evaluate.generation_record({"prediction": "alpha clause", "answer": "alpha clause"})
    assert record["gen_char_f1"] == pytest.approx(1.0)
    assert record["gen_token_f1"] == pytest.approx(1.0)
    assert record["gen_em"] == 1.0


def test_generation_record_skipped_without_prediction_or_answer():
    assert evaluate.generation_record({"prediction": "", "answer": "x"}) == {}
    assert evaluate.generation_record({"prediction": "x", "answer": ""}) == {}


def test_aggregate_scales_to_percent_and_skips_missing():
    records = [{"acc@5": 1.0, "gen_char_f1": 0.5}, {"acc@5": 0.0}]
    aggregated = evaluate.aggregate(records)
    assert aggregated["acc@5"] == pytest.approx(50.0)
    assert aggregated["gen_char_f1"] == pytest.approx(50.0)  # averaged over defined rows only


def test_common_helpers_defaults():
    assert common.method_names(MINIMAL_CONFIG) == ["ours", "bm25"]
    assert common.top_k(MINIMAL_CONFIG) == 3
    assert common.evaluation_section(MINIMAL_CONFIG)["k"] == [5]


def test_lazy_encoder_defers_model_loading():
    encoder = common.build_encoder(MINIMAL_CONFIG)
    assert encoder.loaded is False  # no model touched until something embeds


def test_code_version_fingerprint_is_stable():
    first = common.code_version()
    second = common.code_version()
    assert first["code_fingerprint"] == second["code_fingerprint"]
    assert len(first["code_fingerprint"]) == 40
    assert "python" in first and "numpy" in first