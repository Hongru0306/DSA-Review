"""GraphRAG baseline adapter (microsoft/graphrag).

Wraps the official GraphRAG package behind our ``Retriever`` interface:
- indexing runs the GraphRAG pipeline over the corpus (each doc tagged with a
  ``[DOCID:i]`` marker), in ``build``;
- retrieval runs local search and parses retrieved source text units back to
  corpus indices, in ``rank``.

Requires the optional dependency ``graphrag`` (see ``requirements-baselines.txt``)
plus an OpenAI-compatible chat and embedding endpoint. GraphRAG writes its index
to ``workspace`` and reloads it on construction if already present.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from retrieval.base import Retriever
from retrieval._support import parse_doc_ids, require, tag_docs

_INSTALL_HINT = "Install it with `pip install graphrag` (see requirements-baselines.txt)."

_TABLE_NAMES = ("entities", "communities", "community_reports", "text_units", "relationships")


class GraphRAGRetriever(Retriever):
    name = "GraphRAG"

    def __init__(
        self,
        workspace: str,
        llm_model: str,
        embedding_model: str,
        api_base: Optional[str] = None,
        api_key: str = "",
        embedding_api_base: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        community_level: int = 2,
        response_type: str = "Single Sentence",
        top_k: int = 10,
    ):
        self.workspace = Path(workspace)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.api_base = api_base
        self.api_key = api_key
        self.embedding_api_base = embedding_api_base or api_base
        self.embedding_api_key = embedding_api_key or api_key
        self.community_level = community_level
        self.response_type = response_type
        self.top_k = top_k
        self.n_docs = 0
        self._config = None

    # ---- Index construction ----

    def build(self, corpus: List[str], encoder) -> None:
        graphrag = require("graphrag", _INSTALL_HINT)
        self.n_docs = len(corpus)
        input_dir = self.workspace / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "corpus.txt").write_text("\n".join(tag_docs(corpus)), encoding="utf-8")
        self._write_settings()
        subprocess.run(
            [sys.executable, "-m", "graphrag", "index", "--root", str(self.workspace)],
            check=True,
        )
        self._config = graphrag.config.load_config(str(self.workspace))

    def _write_settings(self) -> None:
        """Generate the GraphRAG settings file (via `graphrag init`) and point its models at our endpoints."""
        yaml = require("yaml", "PyYAML is required by graphrag.")
        settings_path = self.workspace / "settings.yaml"
        if not settings_path.exists():
            subprocess.run(
                [sys.executable, "-m", "graphrag", "init", "--root", str(self.workspace)],
                check=True,
            )
        settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        models = settings.setdefault("models", {})
        models["default_chat_model"] = {
            "type": "openai_chat",
            "model": self.llm_model,
            "api_base": self.api_base,
            "api_key": self.api_key,
        }
        models["default_embedding_model"] = {
            "type": "openai_embedding",
            "model": self.embedding_model,
            "api_base": self.embedding_api_base,
            "api_key": self.embedding_api_key,
        }
        settings_path.write_text(
            yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    # ---- Retrieval ----

    def rank(self, q: str) -> List[int]:
        graphrag = require("graphrag", _INSTALL_HINT)
        if self._config is None:
            self._config = graphrag.config.load_config(str(self.workspace))
        import pandas as pd

        output_dir = self.workspace / "output"
        tables = {name: pd.read_parquet(output_dir / f"{name}.parquet") for name in _TABLE_NAMES}
        _response, context = asyncio.run(
            graphrag.api.local_search(
                self._config,
                entities=tables["entities"],
                communities=tables["communities"],
                community_reports=tables["community_reports"],
                text_units=tables["text_units"],
                relationships=tables["relationships"],
                covariates=None,
                community_level=self.community_level,
                response_type=self.response_type,
                query=q,
            )
        )
        text = json.dumps(context, ensure_ascii=False, default=str)
        ids = parse_doc_ids(text, self.n_docs)
        return ids if ids else list(range(self.n_docs))