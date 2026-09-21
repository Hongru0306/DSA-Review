"""Factory to build retrievers by name, bridging this package's LLMClient / SemanticEncoder.

The real-RAG adapters each expect framework-specific hooks (callables, model
names, or HTTP endpoints). This factory maps a single ``(encoder, llm_client)``
pair onto every adapter, so callers do not have to wire the details by hand.

Adapters whose framework is not installed still construct here; they only raise
a clear ``ImportError`` when ``build`` / ``rank`` is called.

Example:
    from encoder import SemanticEncoder
    from llm import LLMClient
    from retrieval.factory import build_retriever

    encoder = SemanticEncoder()
    llm = LLMClient(api_key=..., base_url="https://api.deepseek.com", model="deepseek-v4-flash")
    retriever = build_retriever("lightrag", encoder=encoder, llm_client=llm, working_dir="runs/lightrag")
    retriever.build(corpus, encoder)
    ranking = retriever.rank("养护时间不应少于14天")
"""

from __future__ import annotations

from typing import Any, Optional

from retrieval.base import Retriever
from retrieval.bm25 import BM25
from retrieval.dense import DenseNaive
from retrieval.graphrag import GraphRAGRetriever
from retrieval.hipporag import HippoRAGRetriever
from retrieval.lightrag import (
    LightRAGRetriever,
    async_embedding_from_encoder,
    async_llm_from_client,
)
from retrieval.ours import OursRetriever
from retrieval.raptor import RaptorRetriever

_ALIASES = {
    "ours": "ours",
    "ours-no-rel": "ours_no_rel",
    "bm25": "bm25",
    "naive": "dense",
    "naiverag": "dense",
    "dense": "dense",
    "graphrag": "graphrag",
    "lightrag": "lightrag",
    "hipporag": "hipporag",
    "raptor": "raptor",
}

RETRIEVER_NAMES = tuple(sorted(_ALIASES))


def _llm_model(llm_client: Any) -> Optional[str]:
    return getattr(llm_client, "model", None)


def _llm_base(llm_client: Any) -> Optional[str]:
    return getattr(llm_client, "base_url", None) or getattr(llm_client, "endpoint", None)


def _embedding_model(encoder: Any) -> Optional[str]:
    return getattr(encoder, "model_name", None)


def build_retriever(
    name: str,
    *,
    encoder: Any = None,
    llm_client: Any = None,
    **kwargs: Any,
) -> Retriever:
    """Construct a retriever by name, injecting the bridged LLM / embedding hooks.

    ``encoder`` is any object with ``encode_batch`` (e.g. ``SemanticEncoder``);
    ``llm_client`` is any object with ``call_text`` (e.g. ``LLMClient``). Any
    extra ``kwargs`` are forwarded to the adapter constructor.
    """
    key = _ALIASES.get(str(name).strip().lower())
    if key is None:
        raise ValueError(f"Unknown retriever '{name}'. Known names: {', '.join(RETRIEVER_NAMES)}")

    if key == "ours":
        return OursRetriever(**kwargs)
    if key == "ours_no_rel":
        kwargs.setdefault("use_rel", False)
        return OursRetriever(**kwargs)
    if key == "bm25":
        return BM25(**kwargs)
    if key == "dense":
        return DenseNaive(**kwargs)

    if key == "lightrag":
        if llm_client is not None:
            kwargs.setdefault("llm_model_func", async_llm_from_client(llm_client))
        if encoder is not None:
            kwargs.setdefault("embedding_func", async_embedding_from_encoder(encoder))
        return LightRAGRetriever(**kwargs)

    if key == "raptor":
        if llm_client is not None:
            kwargs.setdefault("llm_client", llm_client)
        return RaptorRetriever(**kwargs)

    if key == "hipporag":
        kwargs.setdefault("llm_model_name", _llm_model(llm_client))
        kwargs.setdefault("llm_base_url", _llm_base(llm_client))
        kwargs.setdefault("embedding_model_name", _embedding_model(encoder))
        return HippoRAGRetriever(**kwargs)

    if key == "graphrag":
        # GraphRAG indexes and queries through HTTP endpoints, so it needs a *base
        # URL* for both chat and embeddings; an in-process encoder cannot be used.
        # Pass embedding_api_base / embedding_model explicitly for the embedding side.
        kwargs.setdefault("llm_model", _llm_model(llm_client))
        kwargs.setdefault("api_base", _llm_base(llm_client))
        kwargs.setdefault("api_key", getattr(llm_client, "api_key", ""))
        if encoder is not None:
            kwargs.setdefault("embedding_model", _embedding_model(encoder))
        return GraphRAGRetriever(**kwargs)

    raise AssertionError(f"unhandled retriever key: {key}")