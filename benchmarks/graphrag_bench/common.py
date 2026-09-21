"""Shared helpers for the GraphRAG-Bench prepare / infer / evaluate pipeline.

Everything the three CLIs need lives here: config loading, prepared-artifact IO,
lazy encoder / LLM construction, corpus-graph construction, and retriever
construction (through ``retrieval.factory``, which bridges the LLM client and
encoder into each adapter).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from graph import build_corpus_graph
from retrieval import build_retriever
from utils.io import read_jsonl, write_json, write_jsonl

DEFAULT_CONFIG = Path("configs/graphrag_bench.yaml")
PREPARED_CORPUS = "corpus.jsonl"
PREPARED_QUERIES = "queries.jsonl"
PREPARED_MANIFEST = "manifest.json"

# Files that make up a code version fingerprint (relative to the repository root).
_FINGERPRINT_DIRS = ("retrieval", "graph", "generation", "evaluation", "llm", "utils", "benchmarks")
_FINGERPRINT_FILES = ("config.py", "encoder.py")


# ---- config ----


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"config not found: {config_path}. Pass --config or create it from configs/graphrag_bench.yaml."
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config["_config_path"] = str(config_path)
    return config


def resolve_dir(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a config-relative path against the current working directory."""
    return Path(value)


def data_section(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("data", {}) or {}


def prepared_dir(config: dict[str, Any]) -> Path:
    return resolve_dir(config, data_section(config).get("prepared_dir", "outputs/graphrag_bench/prepared"))


def default_output_dir(config: dict[str, Any]) -> Path:
    return resolve_dir(config, data_section(config).get("output_dir", "outputs/graphrag_bench"))


def retrieval_section(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("retrieval", {}) or {}


def evaluation_section(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("evaluation", {}) or {}


def generation_section(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("generation", {}) or {}


def models_section(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("models", {}) or {}


def method_names(config: dict[str, Any]) -> list[str]:
    methods = retrieval_section(config).get("methods") or ["ours", "bm25", "naive"]
    return [str(m).strip().lower() for m in methods if str(m).strip()]


def top_k(config: dict[str, Any]) -> int:
    return int(retrieval_section(config).get("top_k", 5))


# ---- prepared artifacts ----


def corpus_file(config: dict[str, Any]) -> Path:
    return prepared_dir(config) / PREPARED_CORPUS


def queries_file(config: dict[str, Any]) -> Path:
    return prepared_dir(config) / PREPARED_QUERIES


def require_prepared(config: dict[str, Any]) -> None:
    missing = [p for p in (corpus_file(config), queries_file(config)) if not p.exists()]
    if missing:
        raise SystemExit(
            "prepared benchmark data not found:\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\nRun `python -X utf8 -m benchmarks.graphrag_bench.prepare --config <config>` first."
        )


def load_corpus(config: dict[str, Any]) -> list[dict[str, Any]]:
    return read_jsonl(corpus_file(config))


def load_queries(config: dict[str, Any]) -> list[dict[str, Any]]:
    return read_jsonl(queries_file(config))


def write_prepared(
    config: dict[str, Any],
    corpus: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    write_jsonl(corpus_file(config), corpus)
    write_jsonl(queries_file(config), queries)
    write_json(prepared_dir(config) / PREPARED_MANIFEST, manifest)


# ---- models ----


class LazyEncoder:
    """Encoder proxy that loads the embedding model on first use.

    Retrievers that never embed (e.g. BM25) stay importable and runnable without
    an embedding model present.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name
        self._encoder = None

    def _load(self) -> Any:
        if self._encoder is None:
            from encoder import SemanticEncoder

            self._encoder = SemanticEncoder(model_name=self.model_name)
        return self._encoder

    def encode_batch(self, texts):
        return self._load().encode_batch(texts)

    def encode_single(self, text):
        return self._load().encode_single(text)

    @property
    def loaded(self) -> bool:
        return self._encoder is not None


def build_encoder(config: dict[str, Any]) -> LazyEncoder:
    env_name = models_section(config).get("embedding", {}).get("model_env", "EMBEDDING_MODEL")
    return LazyEncoder(model_name=os.environ.get(env_name) or None)


def build_llm(config: dict[str, Any], required: bool = False) -> Any:
    llm_cfg = models_section(config).get("llm", {}) or {}
    api_key = os.environ.get(llm_cfg.get("api_key_env", "LLM_API_KEY"), "").strip()
    if not api_key:
        if required:
            raise SystemExit(
                f"{llm_cfg.get('api_key_env', 'LLM_API_KEY')} is not set; "
                "generation requires a configured LLM endpoint (see README)."
            )
        return None
    from llm import LLMClient

    return LLMClient(
        api_key=api_key,
        base_url=os.environ.get(llm_cfg.get("base_url_env", "LLM_BASE_URL"), "")
        or llm_cfg.get("default_base_url", "https://api.deepseek.com"),
        model=os.environ.get(llm_cfg.get("model_env", "LLM_MODEL"), "")
        or llm_cfg.get("default_model", "deepseek-v4-flash"),
    )


# ---- corpus graph ----


def build_graph(config: dict[str, Any], corpus: Iterable[str]) -> dict[str, Any] | None:
    corpus_cfg = config.get("corpus", {}) or {}
    mode = str(corpus_cfg.get("build_graph", "deterministic")).lower()
    if mode in {"none", "null", "off"}:
        return None
    cache = corpus_cfg.get("graph_cache")
    if mode == "cache" or cache:
        if not cache:
            raise SystemExit("corpus.graph_cache must be set when corpus.build_graph is 'cache'.")
        from utils.io import read_json

        return read_json(cache)
    texts = list(corpus)
    return build_corpus_graph(
        texts,
        max_entities=int(corpus_cfg.get("max_entities", 24)),
        max_relations=int(corpus_cfg.get("max_relations", 48)),
    )


# ---- retrievers ----


def _adapter_kwargs(config: dict[str, Any], method: str) -> dict[str, Any]:
    adapters = retrieval_section(config).get("adapters", {}) or {}
    kwargs = dict(adapters.get(method, {}) or {})
    if method == "ours":
        kwargs.update(retrieval_section(config).get("ours", {}) or {})
    return kwargs


def build_method(
    method: str,
    config: dict[str, Any],
    encoder: Any,
    llm_client: Any,
    corpus: list[str],
    corpus_graph: dict[str, Any] | None,
) -> Any:
    """Construct and ``build`` the requested retriever."""
    kwargs = _adapter_kwargs(config, method)
    retriever = build_retriever(method, encoder=encoder, llm_client=llm_client, **kwargs)
    if method == "ours":
        retriever.build(corpus, encoder, corpus_graph=corpus_graph)
    else:
        retriever.build(corpus, encoder)
    return retriever


# ---- versions ----


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha1()
    paths: list[Path] = [root / name for name in _FINGERPRINT_FILES]
    for directory in _FINGERPRINT_DIRS:
        paths.extend(sorted((root / directory).rglob("*.py")))
    for path in sorted({p for p in paths if p.exists()}):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def code_version() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return {
        "code_fingerprint": _fingerprint(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def data_version(corpus: list[dict[str, Any]], queries: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha1()
    for row in corpus:
        digest.update(str(row.get("text", "")).encode("utf-8"))
    for row in queries:
        digest.update(str(row.get("question", "")).encode("utf-8"))
    return {
        "corpus_rows": len(corpus),
        "query_rows": len(queries),
        "data_fingerprint": digest.hexdigest(),
    }


def run_config_snapshot(config: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    llm_cfg = models_section(config).get("llm", {}) or {}
    emb_cfg = models_section(config).get("embedding", {}) or {}
    snapshot = {
        "config": {k: v for k, v in config.items() if not str(k).startswith("_")},
        "config_path": config.get("_config_path"),
        "models": {
            "llm_model": os.environ.get(llm_cfg.get("model_env", "LLM_MODEL"))
            or llm_cfg.get("default_model"),
            "llm_base_url": os.environ.get(llm_cfg.get("base_url_env", "LLM_BASE_URL"), ""),
            "embedding_model": os.environ.get(emb_cfg.get("model_env", "EMBEDDING_MODEL")),
        },
        "versions": code_version(),
    }
    if extra:
        snapshot.update(extra)
    return snapshot


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
