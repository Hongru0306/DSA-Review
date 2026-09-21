# DSA-Review

Implementation of **No Passage Is an Island: Dual-Subgraph Alignment for Context-Aware Construction Document Review**.

This repository provides the DSA-Review inference pipeline, scripts for running inference and evaluation on benchmark, and runnable examples of the document review workflow.

## Overview

DSA-Review retrieves supporting evidence by aligning context-enriched query subgraphs with subgraphs constructed from the reference corpus.

**Context-Enriched Subgraph Fusion (CESF)** incorporates surrounding context into passage representations and fuses the resulting entity and relation subgraphs into a shared knowledge structure.

**Query Subgraph Alignment and Retrieval (QSAR)** constructs a query subgraph, identifies candidate regions in the corpus graph, and ranks supporting passages through subgraph alignment. The retrieved evidence is then used for answer generation or document review.

## Installation

```bash
git clone https://github.com/Hongru0306/DSA-Review.git
cd DSA-Review
python -m pip install -r requirements.txt
```

Run the commands below from the repository root. Install the additional dependencies specified by the selected baseline configuration when running external baseline frameworks.

## Quick Start

Run the lightweight retrieval example:

```bash
python -X utf8 examples/demo.py
```

The example uses a small synthetic corpus and a deterministic test encoder to demonstrate retrieval and metric computation with BM25 and dense retrieval. It does not require model downloads or an LLM endpoint. The printed scores are example outputs, not benchmark results.

Time: Installation typically takes less than 30 minutes, depending on network speed.

## Model Configuration

For model-backed inference, configure the LLM endpoint and embedding model. Use the model identifiers specified by the experiment configuration.

```bash
export LLM_API_KEY="YOUR_API_KEY"
export LLM_BASE_URL="YOUR_LLM_ENDPOINT"
export LLM_MODEL="YOUR_LLM_MODEL_ID"
export EMBEDDING_MODEL="YOUR_EMBEDDING_MODEL_ID_OR_LOCAL_PATH"
```

In PowerShell, assign these variables using `$env:VARIABLE_NAME = "value"`.

## Document Review Example

After configuring the models, run the example review workflow:

```bash
python -X utf8 examples/review_demo.py \
  --input examples/review_example.json \
  --output outputs/example/review.json
```

The example demonstrates how a passage and its surrounding context are processed to retrieve supporting clauses and produce a review result. The output records the retrieved evidence, assessment, and suggested revision.

## GraphRAG-Bench

The public benchmark data are available from the [GraphRAG-Bench dataset repository](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench).

The experiment configuration in `configs/graphrag_bench.yaml` specifies the evaluated subset, input locations, model settings, retrieval parameters, and evaluation settings. See `docs/benchmark_protocol.md` for corpus preparation, evaluation sample selection, metric definitions, and aggregation rules.

### Prepare the benchmark

```bash
python -X utf8 -m benchmarks.graphrag_bench.prepare \
  --config configs/graphrag_bench.yaml
```

### Run inference

```bash
python -X utf8 -m benchmarks.graphrag_bench.infer \
  --config configs/graphrag_bench.yaml \
  --output-dir outputs/graphrag_bench
```

### Evaluate predictions

```bash
python -X utf8 -m benchmarks.graphrag_bench.evaluate \
  --config configs/graphrag_bench.yaml \
  --predictions outputs/graphrag_bench/predictions.jsonl \
  --output outputs/graphrag_bench/metrics.json
```

Inference and evaluation are separate steps, allowing saved predictions to be rescored without repeating answer generation.

### Outputs

```text
outputs/graphrag_bench/
├── predictions.jsonl
├── metrics.json
└── run_config.json
```

`predictions.jsonl` stores per-query predictions and retrieved evidence. `metrics.json` contains aggregate evaluation results. `run_config.json` records the resolved experiment settings, model identifiers, and code and data versions.

To rerun an experiment, use the corresponding configuration and the same benchmark version and evaluation protocol. Model-backed outputs may vary across repeated runs or changes to the model service.

## Repository Structure

```text
benchmarks/     Public benchmark preparation, inference, and evaluation
configs/        Experiment configurations
docs/           Benchmark protocol and implementation documentation
graph/          Corpus and query graph construction
retrieval/      DSA-Review retrieval and baseline adapters
generation/     Evidence-based answer generation and document review
evaluation/     Retrieval, generation, and review metrics
llm/            LLM client
encoder.py      Text embedding interface
examples/       Runnable examples
tests/          Unit tests
utils/          Shared utilities
```

## Testing

```bash
python -m pytest tests -q
```

Unit tests check retrieval behavior and metric computation on small synthetic inputs. Public benchmark results are produced through the inference and evaluation workflow above.

## Citation

BibTeX will be added upon acceptance.

## License and Usage Restrictions

This repository is provided solely for peer review and reproduction of the results reported in the accompanying manuscript.

Prior to acceptance of the manuscript, redistribution, public dissemination, and use for further development or derivative works are prohibited. Modifications are permitted only as necessary to reproduce and verify the reported results during peer review.

The license governing subsequent use will be announced upon acceptance. Acceptance alone does not grant additional usage rights.
