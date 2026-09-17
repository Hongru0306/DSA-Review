"""LightRAG baseline adapter (HKUDS/LightRAG).

Wraps ``lightrag.LightRAG`` behind our ``Retriever`` interface: documents (each
tagged with ``[DOCID:i]``) are inserted in ``build``; ``rank`` runs a query with
``only_need_context=True`` and parses the returned context back to corpus
indices. The LLM and embedding backends are injected as callables; helper
factories bridge this package's ``LLMClient`` / ``SemanticEncoder`` to them.

Requires the optional dependency ``lightrag-hku`` plus an LLM and embedding endpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional

from retrieval.base import Retriever
from retrieval._support import parse_doc_ids, require, tag_docs

_INSTALL_HINT = "Install it with `pip install lightrag-hku` (see requirements-baselines.txt)."


def async_llm_from_client(client: Any) -> Callable:
    """Adapt an ``LLMClient``-like object (with ``call_text``) to LightRAG's async llm_model_func."""

    async def _fn(prompt, system_prompt=None, history_messages=None, **kwargs):
        return client.call_text(system_prompt or "", prompt)

    return _fn


def async_embedding_from_encoder(encoder: Any) -> Callable:
    """Adapt a ``SemanticEncoder``-like object (with ``encode_batch``) to LightRAG's async embedding_func."""

    async def _fn(texts, context=None):
        return encoder.encode_batch(list(texts))

    return _fn


class LightRAGRetriever(Retriever):
    name = "LightRAG"

    def __init__(
        self,
        working_dir: str,
        llm_model_func: Optional[Callable] = None,
        embedding_func: Optional[Callable] = None,
        mode: str = "hybrid",
        top_k: int = 10,
        chunk_token_size: int = 1200,
        chunk_overlap_token_size: int = 100,
    ):
        self.working_dir = Path(working_dir)
        self.llm_model_func = llm_model_func
        self.embedding_func = embedding_func
        self.mode = mode
        self.top_k = top_k
        self.chunk_token_size = chunk_token_size
        self.chunk_overlap_token_size = chunk_overlap_token_size
        self.n_docs = 0
        self._rag = None

    def build(self, corpus: List[str], encoder) -> None:
        lightrag = require("lightrag", _INSTALL_HINT)
        from lightrag.utils import EmbeddingFunc

        if self.llm_model_func is None:
            raise ValueError("LightRAGRetriever requires llm_model_func (see async_llm_from_client).")
        embed_callable = self.embedding_func or async_embedding_from_encoder(encoder)
        embedding_dim = int(encoder.encode_batch(["_dim_probe_"]).shape[1])
        embedding = EmbeddingFunc(
            embedding_dim=embedding_dim, max_token_size=8192, func=embed_callable
        )
        self._rag = lightrag.LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=self.llm_model_func,
            embedding_func=embedding,
            chunk_token_size=self.chunk_token_size,
            chunk_overlap_token_size=self.chunk_overlap_token_size,
        )
        self.n_docs = len(corpus)
        self._rag.insert(tag_docs(corpus))

    def rank(self, q: str) -> List[int]:
        require("lightrag", _INSTALL_HINT)
        from lightrag import QueryParam

        context = self._rag.query(
            q,
            param=QueryParam(mode=self.mode, top_k=self.top_k, only_need_context=True),
        )
        ids = parse_doc_ids(context, self.n_docs)
        return ids if ids else list(range(self.n_docs))