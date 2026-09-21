"""GraphRAG-Bench preparation, inference, and evaluation.

CLIs:
    python -X utf8 -m benchmarks.graphrag_bench.prepare  --config configs/graphrag_bench.yaml
    python -X utf8 -m benchmarks.graphrag_bench.infer    --config configs/graphrag_bench.yaml --output-dir outputs/graphrag_bench
    python -X utf8 -m benchmarks.graphrag_bench.evaluate --config configs/graphrag_bench.yaml \
        --predictions outputs/graphrag_bench/predictions.jsonl --output outputs/graphrag_bench/metrics.json
"""
