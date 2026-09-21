# Benchmark Protocol

This document defines how the public benchmark is prepared, evaluated, and
aggregated. It covers the `benchmarks/graphrag_bench` suite:

```text
prepare   ->  <prepared_dir>/{corpus.jsonl, queries.jsonl, manifest.json}
infer     ->  <output_dir>/{predictions.jsonl, run_config.json}
evaluate  ->  <output_dir>/metrics.json
```

All metric implementations referenced below live in `evaluation/`; the numbers
reported in the paper are produced by this pipeline with the configuration in
`configs/graphrag_bench.yaml`.

## 1. Benchmark source

The public data are the [GraphRAG-Bench dataset](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench).
`prepare` supports two sources, selected by `data.source`:

- `local` — read `<data.local_dir>/corpus.jsonl` and `queries.jsonl` already in the
  pipeline schema (input locations are configurable via `data.corpus_file` /
  `data.queries_file`).
- `huggingface` — pull `data.hf_repo` / `data.hf_config` / `data.split` through the
  `datasets` library.

Because fields differ across benchmark releases, the mapping from source fields to
the pipeline schema is configurable through `data.field_map`
(`question`, `answer`, `passages`, `context`, `gold`, `group`, `qid`, `title`,
`source`). `gold` entries may be passage indices or passage texts; texts are
resolved to indices, and unresolvable entries are dropped.

## 2. Corpus preparation

Each reference passage becomes one retrieval unit:

```json
{"doc_id": 0, "text": "...", "source": "...", "title": "..."}
```

- `doc_id` is contiguous (`0..N-1`) and stable within a prepared snapshot, so gold
  indices and every prediction refer to the same unit.
- Retrieval units are the benchmark's own passages; no additional chunking or
  summarisation is applied, so all methods see the same corpus.

The corpus graph used by the DSA-Review retriever is controlled by
`corpus.build_graph`:

- `deterministic` (default) — build the entity/relation graph from the corpus
  itself (`graph.build_corpus_graph`); no LLM, no gold labels, fully reproducible.
- `cache` — load a prebuilt graph from `corpus.graph_cache` (e.g. the LLM-built
  graph cache produced by `graph.llm_builder`).
- `none` — no graph; valid for BM25, dense retrieval, and the framework baselines.

The graph cache is part of the experiment artifact, not of this repository.

## 3. Evaluation sample selection

`data.sample` selects the evaluated subset deterministically:

```yaml
sample:
  size: null        # null = use the whole subset; int = subset size
  seed: 20260101
```

Selection is a seeded random sample of queries (`random.Random(seed)`), returned in
the original query order, so a given `(subset, size, seed)` always yields the same
sample. The chosen sample, the source, and the resulting counts are recorded in
`manifest.json`.

## 4. Retrieval protocol

Each method retrieves for every query in the sample, independently and in a single
shot, and returns a ranking over `doc_id`s. Reported retrieval is at `evaluation.k`.

- `retrieval.methods` selects the methods; all methods use the same
  `retrieval.top_k` and the same corpus.
- DSA-Review parameters are set under `retrieval.ours` (`n`, `m`, `gamma`,
  `alpha`, `lambda_`).
- Framework baselines take their own constructor arguments under
  `retrieval.adapters` (state directories, query mode, community level, ...).
- Framework baselines are optional dependencies (see `requirements-baselines.txt`);
  a method whose framework is not installed raises a clear `ImportError` instead of
  silently degrading.

## 5. Generation protocol

When `generation.enabled` is true and an LLM endpoint is configured, each query is
answered from the top-`generation.top_k` retrieved passages:

- The retrieved passages are formatted as a numbered evidence block; the model is
  instructed to quote the supporting passage verbatim and to give a short
  extractive answer (`generation.qa.build_answer_prompt`).
- Answers are validated (length bounds, no refusal, every quoted span is a verbatim
  substring of a retrieved passage); invalid answers are retried with the
  validation error, and finally repaired from the retrieved evidence only
  (`generation.qa.repair_prediction`). Gold answers are never used during
  generation.
- A generation failure is recorded per query (`generation_error`) and does not
  abort the run.

## 6. Metric definitions

Per query, with `top-k_q` the ranked predictions, `gold_q` the set of gold
`doc_id`s, and `rel_i` the relevance indicator at rank `i`:

| Metric | Definition |
|---|---|
| `acc@k` | `1[gold_q ∩ top-k_q ≠ ∅]` (binary hit; `retrieval_metrics["Hit@k"]`) |
| `coverage@k` | `\|gold_q ∩ top-k_q\| / \|gold_q\|` (evidence coverage; equals `Recall@k`) |
| `precision@k` | `\|gold_q ∩ top-k_q\| / k` |
| `recall@k` | alias of `coverage@k` |
| `mrr@k` | `1/rank` of the first relevant prediction within `k`, else `0` |
| `ndcg@k` | `DCG@k / IDCG@k` with `DCG@k = Σ_i rel_i / log2(i+2)` and `IDCG@k = Σ_{i=1..min(\|gold\|,k)} 1/log2(i+2)` |
| `map@k` | `(1/min(\|gold\|,k)) · Σ_{i: rel_i} (rank_i-th hit count)/i` |
| `ret_avg@k` | `(acc@k + mrr@k + ndcg@k) / 3` |

Generation metrics (`evaluation/generation.py`):

| Metric | Definition |
|---|---|
| `gen_char_precision/recall/f1` | multiset character overlap after whitespace removal (`char_prf`) |
| `gen_token_precision/recall/f1` | multiset token overlap after SQuAD-style normalisation — lowercase, strip punctuation, drop `a/an/the` (`token_prf`) |
| `gen_em` | `1` iff the normalised prediction equals the normalised reference (`exact_match`) |

`Gen.Avg = (gen_char_f1 + gen_token_f1) / 2`. `gen_em` is reported alongside and is
deliberately not averaged into `Gen.Avg`, because it is a strict 0/1 indicator.

For the document-review workflow the same protocol applies with review-oriented
metrics (`evaluation/revision.py`): character PRF over the **edit payload** relative
to the audited sentence (`revision_diff_char_prf`), a BERTScore-style embedding PRF
over revised chunks (`embedding_prf`), and a delta-embedding PRF over the edit
direction (`delta_embedding_prf`). These are demonstrated by
`examples/review_demo.py`.

## 7. Aggregation rules

- Per-query metrics are macro-averaged over queries and reported as percentages
  (`mean × 100`; `evaluation/aggregate.py`).
- A metric is skipped for a query when it is undefined (e.g. a query without a
  prediction contributes no generation metric), and averages are taken over the
  queries where it is defined.
- `evaluation.group_by` additionally reports every metric per group (for example
  `hop` for multi-hop subsets). Groups use the same definitions and the same
  macro-average.
- Retrieval and generation are aggregated independently: retrieval is scored from
  the saved rankings, generation from the saved predictions.

## 8. Outputs and reproducibility

```text
<prepared_dir>/  corpus.jsonl  queries.jsonl  manifest.json
<output_dir>/    predictions.jsonl  run_config.json  metrics.json
```

- `predictions.jsonl` — one row per (method, query): rankings, retrieved evidence,
  and the generated answer when generation is enabled.
- `run_config.json` — resolved configuration, model identifiers, method list, and
  code/data fingerprints.
- `metrics.json` — per-method macro-averaged metrics, overall and per group.
- `manifest.json` — source, counts, sample settings, and versions.

Reproducibility notes:

- Corpus construction, sample selection, retrieval, and scoring are deterministic
  given the same prepared snapshot and configuration; the code and data
  fingerprints in `run_config.json` identify the exact snapshot used.
- Answer generation depends on the model service; repeated runs or a changed model
  service can change generation metrics and any downstream numbers. Report the
  model identifiers recorded in `run_config.json` alongside results.
- Because inference and evaluation are separate steps, predictions can be rescored
  with the same or a corrected protocol without regenerating answers.
