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
- [Code Provenance](#code-provenance)
- [What Is Not Included](#what-is-not-included)

---

## Module Structure

| Module | Contents |
|---|---|
| Graph construction `graph` | Corpus segmentation → entity / relation graph (deterministic, no LLM; or LLM-incremental caching); query entity / relation parsing |
| Retrieval `retrieval` | **Ours** hypergraph subgraph retrieval (paper Alg. 1 + 2) + baselines GraphRAG-lite / HippoRAG-lite / LightRAG-lite / RAPTOR-lite / Dense BGE / BM25 |
| Generation `generation` | Evidence-grounded QA (evidence → answer, v42 citation / length calibration + validation and repair); clause-attributed review (paper Alg. 3); centralized prompts |
| Evaluation `evaluation` | Retrieval Hit/Recall/Precision/MAP/MRR/nDCG/coverage; generation Char/Token F1, EM, ROUGE-L, BLEU1; revision diff / BERTScore-style embedding F1; aggregation |
| LLM `llm` | Unified DeepSeek / Qwen-vLLM client (thinking disabled, JSON mode, retries, JSON repair) |

Modules are imported as top-level packages. The repository root is the source root.

---

## Installation and Import

```bash
pip install -r requirements.txt        # or pip install -e .
cd <repo>
python -c "from retrieval.ours import OursRetriever"
```

---

## Quick Start

Offline demo. No data, models, or network required.

```bash
PYTHONIOENCODING=utf-8 python examples/demo.py
```

Synthesizes a 5-clause corpus, builds the graph deterministically, and runs all methods plus retrieval metrics.

---

## Testing

```bash
pytest -q
```

All tests use injected fake encoders and a tiny synthetic corpus, so no downloads or network access are needed.

---

## Data Schema

**Input contract**:

- **corpus**: `List[str]`, each entry is a clause / corpus chunk of text.
- **corpus_graph (optional, offline graph artifact)**: `Dict[str, Dict]`, keyed by `sha1(norm(text))` (see `utils.text.doc_cache_key`), with values `{"entities": [...], "relations": [["A","B"], ...], "llm_ok": bool}`. Build with `graph.build_corpus_graph(corpus)` (no LLM) or `graph.build_corpus_graph_cache(...)` (LLM).
- **qcache (optional, query-side LLM cache)**: `Dict[str, Dict]`, `{question: {"entities": [...], "relations": [...]}}`; build with `graph.build_query_cache(questions, path, llm, concurrent)`.
- **Retrieval row (output / generation input)**: `{"method", "qid", "question", "gold": [doc_idx], "retrieved_top10": [corpus rows, each with doc_id/spec_name/clause/text]}`.

---

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

---

## Code Provenance

| This repository | Source file |
|---|---|
| `retrieval/ours.py` + `ours_torch.py` | `scripts/eval_rag_at5.py` `OursCoverageGeneric` L894–1114; `scripts/ours_torch_accel.py` |
| `retrieval/{dense,bm25,ppr,lightrag_lite,raptor}.py` | `scripts/eval_rag_at5.py` L314–504 |
| `graph/builder.py` | `scripts/build_graphrag_novel_context5_graph_local.py` |
| `graph/llm_builder.py` | `scripts/eval_rag_at5.py` L1172–1253 |
| `generation/qa.py` | `autotune/code/scripts/run_construction_qa_long_generation_v42.py`; `scripts/run_construction_generation_20pct_20260713.py` |
| `generation/review.py` | `scripts/run_construction_review_v2.py` L4440–4533, L3478–3476 |
| `evaluation/*` | `scripts/eval_rag_at5.py` L100–212; `scripts/score_and_emit_tables_20260713.py`; `scripts/run_construction_review_v2.py` L3189–3371 |
| `llm/client.py` | `scripts/run_construction_review_v2.py` L302–628; `scripts/evidence_gated_generate.py` L147–186 |

The GraphRAG / LightRAG baselines in the table are the lite checkpoint implementations used in the official experiments. This repository does not wrap the official `graphrag` / `lightrag` adapters that require external installation packages.

---

## What Is Not Included

- Data files and graph index pkl/jsonl artifacts. Only schemas and examples are defined.
- Weak baselines outside the annotation / preview tasks (Random / SpecMarker / Length flagging heuristics), LLM-aug (HyDE / query2doc / IRCoT / FLARE), and RRF fusion.

---

## Citation

BibTeX will be added upon acceptance.

## License

To be determined.
