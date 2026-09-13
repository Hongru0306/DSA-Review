"""Query-side entity / relation resolution: LLM cache first, else heuristic extraction."""

from __future__ import annotations

from typing import Any

from config import MAX_Q, MAX_REL
from utils.text import extract_terms


def resolve_query_graph(
    question: str,
    qcache: dict[str, dict[str, Any]] | None = None,
    use_llm: bool = True,
    max_q: int = MAX_Q,
    max_rel: int = MAX_REL,
) -> tuple[list[str], list[list[str]]]:
    """Return the (entities, relations) query subgraph.

    ``qcache`` format: ``{question: {"entities": [...], "relations": [["A","B"], ...]}}``
    (produced by ``graph.llm_builder.build_query_cache``). Without a cache or
    without an LLM, falls back to ``extract_terms`` heuristic entities and no relations.
    """
    if use_llm and qcache and question in qcache:
        obj = qcache.get(question) or {}
        entities = obj.get("entities") or extract_terms(question, max_q)
        relations = obj.get("relations") or []
        return entities[:max_q], [r[:2] for r in relations[:max_rel]]
    return extract_terms(question, max_q), []