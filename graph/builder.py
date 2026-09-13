"""确定性离线构图:语料切片 -> 实体 / 关系图,无需 LLM 与标注。

移植 ``scripts/build_graphrag_novel_context5_graph_local.py``:
- 专名短语(``PROPER_RE``)+ 单 token 高 IDF 概念抽取,按
  ``8.0 + 1.5*len(parts) + Sum(log1p(corpus/df))`` 打分取 top-k;
- 同一句内共现的实体两两成关系;兜底相邻实体链。
输出以 ``utils.text.doc_cache_key``(规范空白后 SHA-1)为键,与
``OursRetriever`` 的 ``corpus_graph`` 输入对齐。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from utils.text import (
    GENERIC_CAPITALIZED,
    GRAPH_STOPWORDS,
    SENTENCE_RE,
    PROPER_RE,
    doc_cache_key,
    graph_tokenize,
    normalized_entity,
    normalize_space,
)


def build_document_frequency(corpus: list[str]) -> Counter[str]:
    frequency: Counter[str] = Counter()
    for text in corpus:
        frequency.update(
            set(token for token in graph_tokenize(text) if token not in GRAPH_STOPWORDS)
        )
    return frequency


def candidate_entities(
    text: str,
    document_frequency: Counter[str],
    corpus_size: int,
    max_entities: int,
) -> tuple[list[str], dict[str, set[int]]]:
    """抽取每篇的候选实体及其所在句子下标。"""
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    candidates: dict[str, dict[str, Any]] = {}

    def add(surface: str, score: float, sentence_index: int) -> None:
        display = normalize_space(surface).strip(" .,'’\"-_")
        key = normalized_entity(display)
        if not key or key in GRAPH_STOPWORDS or len(key) < 3:
            return
        current = candidates.get(key)
        if current is None:
            candidates[key] = {
                "surface": display,
                "score": score,
                "sentences": {sentence_index},
            }
        else:
            current["score"] = max(float(current["score"]), score)
            current["sentences"].add(sentence_index)

    for sentence_index, sentence in enumerate(sentences):
        for match in PROPER_RE.finditer(sentence):
            surface = match.group(0)
            parts = graph_tokenize(surface)
            if not parts:
                continue
            if len(parts) == 1 and (
                parts[0] in GENERIC_CAPITALIZED
                or document_frequency[parts[0]] > max(20, corpus_size // 100)
            ):
                continue
            score = 8.0 + 1.5 * len(parts)
            score += sum(
                math_log1p(corpus_size / max(document_frequency[part], 1))
                for part in parts
                if part not in GRAPH_STOPWORDS
            )
            add(surface, score, sentence_index)

        sentence_tokens = graph_tokenize(sentence)
        content = [token for token in sentence_tokens if token not in GRAPH_STOPWORDS]
        for token in set(content):
            df = document_frequency[token]
            if df <= 0 or df > max(120, corpus_size // 30):
                continue
            score = math_log1p(corpus_size / df)
            add(token, score, sentence_index)

    ordered = sorted(
        candidates.items(),
        key=lambda item: (
            -float(item[1]["score"]),
            -len(item[0].split()),
            item[0],
        ),
    )[:max_entities]
    entities = [str(item[1]["surface"]) for item in ordered]
    sentence_membership = {key: set(item["sentences"]) for key, item in ordered}
    return entities, sentence_membership


def relation_pairs(
    entities: list[str],
    sentence_membership: dict[str, set[int]],
    max_relations: int,
) -> list[list[str]]:
    """同句共现实体对;无共现时回退相邻实体链。"""
    keys = [normalized_entity(entity) for entity in entities]
    output: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for left_index, left_key in enumerate(keys):
        for right_index in range(left_index + 1, len(keys)):
            right_key = keys[right_index]
            if not (sentence_membership[left_key] & sentence_membership[right_key]):
                continue
            pair_key = tuple(sorted((left_key, right_key)))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            output.append([entities[left_index], entities[right_index]])
            if len(output) >= max_relations:
                return output
    if not output:
        for index in range(min(len(entities) - 1, max_relations)):
            output.append([entities[index], entities[index + 1]])
    return output


def math_log1p(value: float) -> float:
    """log1p 包装,避免裸 import math 语义混淆(gk 代码直接用 math.log1p)。"""
    import math

    return math.log1p(value)


class DeterministicGraphBuilder:
    """语料 -> ``{cache_key: {"entities": [...], "relations": [[a,b],...], "llm_ok": False}}``。"""

    def __init__(
        self,
        max_entities: int = 24,
        max_relations: int = 48,
    ):
        self.max_entities = max_entities
        self.max_relations = max_relations

    def build(self, corpus: list[str]) -> dict[str, dict[str, Any]]:
        document_frequency = build_document_frequency(corpus)
        graph: dict[str, dict[str, Any]] = {}
        for text in corpus:
            entities, membership = candidate_entities(
                text,
                document_frequency,
                len(corpus),
                self.max_entities,
            )
            relations = relation_pairs(entities, membership, self.max_relations)
            graph[doc_cache_key(text)] = {
                "entities": entities,
                "relations": relations,
                "llm_ok": False,
                "extraction": "deterministic-proper-name-and-high-idf-v1",
            }
        return graph


def build_corpus_graph(
    corpus: list[str],
    max_entities: int = 24,
    max_relations: int = 48,
) -> dict[str, dict[str, Any]]:
    """模块级便捷入口,等价于 ``DeterministicGraphBuilder(...).build(corpus)``。"""
    return DeterministicGraphBuilder(max_entities, max_relations).build(corpus)