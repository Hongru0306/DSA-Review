"""查询侧实体 / 关系解析:优先 LLM 缓存,否则启发式抽取。"""

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
    """返回 (entities, relations) 查询子图。

    ``qcache`` 格式:``{question: {"entities": [...], "relations": [["A","B"], ...]}}``
    (由 ``graph.llm_builder.build_query_cache`` 生成)。无缓存或不用 LLM 时,
    退化为 ``extract_terms`` 启发式实体、空关系。
    """
    if use_llm and qcache and question in qcache:
        obj = qcache.get(question) or {}
        entities = obj.get("entities") or extract_terms(question, max_q)
        relations = obj.get("relations") or []
        return entities[:max_q], [r[:2] for r in relations[:max_rel]]
    return extract_terms(question, max_q), []