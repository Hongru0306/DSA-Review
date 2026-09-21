# DSA-Review

Official implementation of **Dual-Subgraph Alignment for Automated Construction Document Review**.

This repository provides the graph construction, retrieval, generation, and evaluation code for the paper. No data files are included. All inputs are passed as JSON dicts or `list[str]`.

All relevant resources will be released upon acceptance.

---

## Contents

- [Module Structure](#module-structure)
- [Installation and Import](#installation-and-import)
- [Quick Start](#quick-start)
- [Testing](#testing)
- [Data Schema](#data-schema)
- [Retrieval Usage](#retrieval-usage)
- [What Is Not Included](#what-is-not-included)

## Module Structure

    graph/         graph construction
    retrieval/     retrieval methods (ours + baselines)
    generation/    answer and review generation
    evaluation/    retrieval and generation metrics
    llm/           LLM client

Modules are imported as top-level packages. The repository root is the source root.

## Installation and Import

```bash
pip install -r requirements.txt        # or pip install -e .
cd <repo>
python -c "from retrieval.ours import OursRetriever"
```

## Quick Start

Offline demo. No data, models, or network required.

```bash
PYTHONIOENCODING=utf-8 python examples/demo.py
```

Synthesizes a 5-clause corpus, builds the graph deterministically, and runs all methods plus retrieval metrics.

## Testing

```bash
pytest -q
```

All tests use injected fake encoders and a tiny synthetic corpus, so no downloads or network access are needed.

## Data Schema

**Input contract**:

- **corpus**: `List[str]`, each entry is a clause / corpus chunk of text.
- **corpus_graph (optional, offline graph artifact)**: `Dict[str, Dict]`, keyed by `sha1(norm(text))` (see `utils.text.doc_cache_key`), with values `{"entities": [...], "relations": [["A","B"], ...], "llm_ok": bool}`. Build with `graph.build_corpus_graph(corpus)` (no LLM) or `graph.build_corpus_graph_cache(...)` (LLM).
- **qcache (optional, query-side LLM cache)**: `Dict[str, Dict]`, `{question: {"entities": [...], "relations": [...]}}`; build with `graph.build_query_cache(questions, path, llm, concurrent)`.
- **Retrieval row (output / generation input)**: `{"method", "qid", "question", "gold": [doc_idx], "retrieved_top10": [corpus rows, each with doc_id/spec_name/clause/text]}`.

## Retrieval Usage

```python
from retrieval import OursRetriever, BM25
from graph import build_corpus_graph
from evaluation import retrieval_metrics

ours = OursRetriever(n=3, alpha=0.03, lambda_=0.85)
ours.build(corpus, encoder, corpus_graph=build_corpus_graph(corpus))   # encoder: object supporting encode_batch
ranking = ours.rank("养护时间不应少于14天")[:5]
print(retrieval_metrics(ranking, gold=[0], k=5))

bm25 = BM25();  bm25.build(corpus, encoder)
```

`encoder` defaults to `encoder.SemanticEncoder` (BGE). Tests may inject a deterministic fake encoder (see `tests/conftest.py`).
`OursRetrieverTorch` is the torch-vectorized scoring version of Ours (use after `.to_torch(device)`).

## What Is Not Included

- Data files and graph index pkl/jsonl artifacts. Only schemas and examples are defined.
- Weak baselines outside the annotation / preview tasks (Random / SpecMarker / Length flagging heuristics), LLM-aug (HyDE / query2doc / IRCoT / FLARE), and RRF fusion.

## Citation

BibTeX will be added upon acceptance.

## License and Usage Restrictions

This repository is provided solely for peer review and reproduction of the results reported in the accompanying manuscript.

Prior to acceptance of the manuscript, redistribution, public dissemination, and use for further development or derivative works are prohibited. Modifications are permitted only as necessary to reproduce and verify the reported results during peer review.

The license governing subsequent use will be announced upon acceptance. Acceptance alone does not grant additional usage rights.

