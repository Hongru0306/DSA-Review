"""用 LLM 离线构建 corpus_graph 与查询 qcache(支持增量落盘)。

移植 ``scripts/eval_rag_at5.py`` 的 ``build_corpus_graph_cache`` /
``build_ours_query_cache``;``llm`` 需提供 ``async chat(messages, temperature, max_tokens)``。
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from config import MAX_DOC_ENT, MAX_DOC_REL, MAX_Q, MAX_REL
from generation.prompts import CORPUS_GRAPH_PROMPT, ENTITY_PROMPT, REL_PROMPT
from utils.io import write_json
from utils.text import doc_cache_key, extract_terms, normalize_space


def parse_corpus_graph(text: str, fallback_text: str) -> dict[str, Any]:
    """解析 LLM 返回的构图 JSON;失败回退启发式实体。"""
    match = re.search(r"\{[\s\S]*\}", text or "")
    data: dict[str, Any] = {}
    if match:
        try:
            data = json.loads(match.group(0))
        except Exception:
            data = {}
    ents = []
    for x in data.get("entities", []) if isinstance(data, dict) else []:
        x = normalize_space(x)
        if 1 < len(x) < 80:
            ents.append(x)
    rels = []
    for relation in data.get("relations", []) if isinstance(data, dict) else []:
        if isinstance(relation, (list, tuple)) and len(relation) >= 2:
            a, b = normalize_space(relation[0]), normalize_space(relation[1])
            if a and b:
                rels.append([a, b])
    return {
        "entities": list(dict.fromkeys(ents))[:MAX_DOC_ENT]
        or extract_terms(fallback_text, MAX_DOC_ENT),
        "relations": rels[:MAX_DOC_REL],
    }


async def build_corpus_graph_cache(
    corpus: list[str],
    cache_path: Path,
    llm: Any,
    concurrent: int,
) -> dict[str, dict[str, Any]]:
    """逐条条款调用 LLM 抽关系图,写入 ``cache_path``(JSON),返回全量缓存。"""
    cache: dict[str, dict[str, Any]] = (
        json.loads(Path(cache_path).read_text(encoding="utf-8"))
        if Path(cache_path).exists()
        else {}
    )
    todo = [
        (doc_cache_key(text), text)
        for text in corpus
        if doc_cache_key(text) not in cache
    ]
    if not todo:
        return cache
    semaphore = asyncio.Semaphore(concurrent)
    done = 0
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    async def one(key: str, text: str) -> None:
        nonlocal done
        async with semaphore:
            try:
                resp = await asyncio.wait_for(
                    llm.chat(
                        [{"role": "user", "content": CORPUS_GRAPH_PROMPT.format(
                            text=text[:3000], max_terms=MAX_DOC_ENT, max_rels=MAX_DOC_REL
                        )}],
                        temperature=0.0,
                        max_tokens=900,
                    ),
                    timeout=240,
                )
                cache[key] = parse_corpus_graph(resp, text)
                cache[key]["llm_ok"] = True
            except Exception as exc:
                cache[key] = {
                    "entities": extract_terms(text, MAX_DOC_ENT),
                    "relations": [],
                    "llm_ok": False,
                    "error": str(exc),
                }
        done += 1
        if done % 50 == 0 or done == len(todo):
            write_json(cache_path, cache)

    for i in range(0, len(todo), max(concurrent * 8, 1)):
        batch = todo[i : i + max(concurrent * 8, 1)]
        await asyncio.gather(*[one(k, t) for k, t in batch])
    write_json(cache_path, cache)
    return cache


async def build_query_cache(
    questions: list[str],
    cache_path: Path,
    llm: Any,
    concurrent: int,
) -> dict[str, dict[str, Any]]:
    """逐问题用 LLM 抽取实体与关系,写入 ``cache_path``,返回全量 qcache。"""
    cache: dict[str, dict[str, Any]] = (
        json.loads(Path(cache_path).read_text(encoding="utf-8"))
        if Path(cache_path).exists()
        else {}
    )
    todo = [question for question in questions if question not in cache]
    if not todo:
        return cache
    semaphore = asyncio.Semaphore(concurrent)
    done = 0
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    async def one(question: str) -> None:
        nonlocal done
        async with semaphore:
            try:
                ents_txt = await asyncio.wait_for(
                    llm.chat(
                        [{"role": "user", "content": ENTITY_PROMPT.format(
                            question=question, max_terms=MAX_Q
                        )}],
                        temperature=0.0,
                        max_tokens=600,
                    ),
                    timeout=240,
                )
                rels_txt = await asyncio.wait_for(
                    llm.chat(
                        [{"role": "user", "content": REL_PROMPT.format(
                            question=question, max_rels=MAX_REL
                        )}],
                        temperature=0.0,
                        max_tokens=600,
                    ),
                    timeout=240,
                )
                llm_ok, error = True, ""
            except Exception as exc:
                ents_txt, rels_txt = "", ""
                llm_ok, error = False, str(exc)
        ents = []
        for line in ents_txt.splitlines():
            line = re.sub(r"^[\d\.\-、\*\s]+", "", line.strip())
            if 1 < len(line) < 60:
                ents.append(line)
        rels = []
        for line in rels_txt.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2 and parts[0] and parts[1]:
                rels.append(parts[:2])
        cache[question] = {
            "entities": ents[:MAX_Q] or extract_terms(question, MAX_Q),
            "relations": rels[:MAX_REL],
            "llm_ok": llm_ok,
            "error": error,
        }
        done += 1
        if done % 20 == 0 or done == len(todo):
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            write_json(cache_path, cache)

    await asyncio.gather(*[one(q) for q in todo])
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, cache)
    return cache