"""证据问答(Evidence -> Answer)生成与校验 / 修复。

移植 ``run_construction_qa_long_generation_v42.py`` 的 prompt 与
``run_construction_generation_20pct_20260713.py`` 的校验 / 修复循环:
``build_answer_prompt`` -> ``LLMClient.call_text`` -> ``valid_answer`` ->
失败带修正重试(至多 3 次) -> ``repair_prediction`` 检索归因兜底。
"""

from __future__ import annotations

import re
from typing import Any

from generation.prompts import (
    OURS_LENGTH_CALIBRATION,
    QA_SYSTEM,
    REFUSALS,
    V42_ANSWER_TEMPLATE,
)
from utils.text import normalized_span, quoted_spans

_NO_EVIDENCE = "（本方法不提供检索证据，请依据题目背景给出最可能的规范结论。）"
_FALLBACK_PROMPT = """你是建筑工程规范问答助手。请回答题目，必须给出明确结论，不得拒答。

输出要求：
1. 只输出答案正文，不输出 Markdown、列表、思考过程、规范名称、规范编号或条款号。
2. 有检索证据时，优先逐字引用包含最终数值、公式、限值或判断词的连续原文，并写成“根据“……”明确……”的单段格式。
3. 只保留直接回答问题的必要条件和结论，不添加无关背景。
4. 答案总长度必须为 101–300 个 Unicode 字符；不要少于101字，也不要超过300字。
5. 即使证据不完整，也必须基于最相关内容给出保守而具体的答案。
{correction}

【方法】{method}
【题目】
{question}

【检索证据】
{evidence}
"""


def split_question(question: str) -> tuple[str, str]:
    """把题目按 ``\\n\\n`` 拆成 (上下文背景, 问题);拆不开时上下文为空。"""
    normalized = question.replace("\r\n", "\n").strip()
    parts = normalized.split("\n\n", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", normalized


def clause_group(spec_id: str, clause: str) -> str:
    parts = re.findall(r"\d+", clause or "")
    if len(parts) >= 2:
        return f"{spec_id}:{parts[0]}.{parts[1]}"
    if parts:
        return f"{spec_id}:{parts[0]}"
    return spec_id


def v42_answer_prompt(row: dict[str, Any], corpus: dict[int, dict], topk: int, extra_instruction: str) -> str:
    """v42 逐字模板(autotune/.../run_construction_qa_long_generation_v42.py build_prompt)。

    注:原代码虽按 gold 文档数准备了三条证据 / 长度细则常量,但未拼入返回模板
    (rules 常量仅作备查),此处保持一致,不额外插值。
    """
    context, query = split_question(row["question"])
    top_hits = row.get("retrieved_top10", [])[:topk]

    group_counts: dict[str, int] = {}
    for hit in top_hits:
        doc = corpus.get(int(hit["doc_id"]), {})
        spec_id = doc.get("spec_id", hit.get("spec_id", ""))
        clause = doc.get("clause", hit.get("clause", ""))
        group = clause_group(spec_id, clause)
        group_counts[group] = group_counts.get(group, 0) + 1

    chunks = []
    for i, hit in enumerate(top_hits, 1):
        doc = corpus.get(int(hit["doc_id"]), {})
        spec_id = doc.get("spec_id", hit.get("spec_id", ""))
        clause = doc.get("clause", hit.get("clause", ""))
        group = clause_group(spec_id, clause)
        chain_tag = "【同小节证据链候选】" if group_counts.get(group, 0) >= 2 else ""
        chunks.append(
            "\n".join(
                [
                    f"[{i}] {chain_tag}{doc.get('spec_name', hit.get('spec_id', ''))} 第{clause}条",
                    doc.get("text", ""),
                ]
            )
        )

    return V42_ANSWER_TEMPLATE.format(
        topk=topk,
        extra_instruction=extra_instruction,
        context=context,
        query=query,
        chunks="\n".join(chunks),
    )


def build_answer_prompt(
    row: dict[str, Any],
    corpus: dict[int, dict],
    method: str,
    correction: str = "",
    top_k: int = 5,
) -> str:
    """方法级 prompt:有检索证据走 v42 模板(校准方法带长度控制),否则回落通用模板。"""
    if row.get("retrieved_top10"):
        method_instruction = correction
        if method in {"Ours", "bge", "jina", "gte", "zhipu", "text-embedding-v3"}:
            method_instruction = (
                OURS_LENGTH_CALIBRATION + ("\n" + correction if correction else "")
            )
        return v42_answer_prompt(row, corpus, topk=top_k, extra_instruction=method_instruction)

    hits = row.get("retrieved_top10", [])[:top_k]
    chunks: list[str] = []
    for index, hit in enumerate(hits, 1):
        doc_id = int(hit["doc_id"] if isinstance(hit, dict) else hit)
        text = corpus.get(doc_id, {}).get("text", "")
        text = text.replace("▼ 展开条文说明", " ").replace("▼", " ")
        text = re.sub(r"\s+", " ", text).strip()
        chunks.append(f"[{index}] {text[:1500]}")
    evidence = "\n".join(chunks) if chunks else _NO_EVIDENCE
    return _FALLBACK_PROMPT.format(
        correction=correction,
        method=method,
        question=row["question"],
        evidence=evidence,
    )


# ---- 校验 ----

def valid_answer(row: dict[str, Any], answer: str, top_k: int = 5) -> tuple[bool, str]:
    length = len(answer)
    if length <= 100:
        return False, f"上一版仅{length}字，请在不改变结论的前提下补充必要适用条件和控制要求，扩写到101–300字。"
    if length > 300:
        return False, f"上一版有{length}字，请删除无关背景并压缩到101–300字。"
    if any(term in answer for term in REFUSALS):
        return False, "上一版出现拒答措辞。不得拒答，请选择最相关证据并给出明确结论。"
    if row.get("retrieved_top10"):
        quotes = quoted_spans(answer)
        if not quotes:
            return False, "上一版没有引文。必须至少逐字引用一个检索条文的连续原文片段。"
        evidence = [
            normalized_span(item.get("text", ""))
            for item in row.get("retrieved_top10", [])[:top_k]
            if isinstance(item, dict)
        ]
        non_verbatim = [
            quote
            for quote in quotes
            if not any(normalized_span(quote) in text for text in evidence)
        ]
        if non_verbatim:
            return False, (
                "上一版引号内容不是检索条文的逐字连续原文，或使用省略号拼接。"
                "请原样复制连续子串，不得改写、概括或省略。"
            )
    return True, ""


# ---- 检索归因兜底修复 ----

def select_retrieval_quote(row: dict[str, Any], draft: str, top_k: int = 5) -> str:
    query = normalized_span(f"{row.get('question', '')}{draft}")
    query_bigrams = {query[i : i + 2] for i in range(max(0, len(query) - 1))}
    candidates: list[tuple[float, str]] = []
    for hit in row.get("retrieved_top10", [])[:top_k]:
        if not isinstance(hit, dict):
            continue
        text = (hit.get("text", "") or "").replace("▼ 展开条文说明", " ").replace("▼", " ")
        text = re.sub(r"\s+", " ", text).strip()
        for segment in re.split(r"(?<=[。；！？])|\|\||\n", text):
            segment = segment.strip(" `#\t")
            if not (4 <= len(segment) <= 140):
                continue
            if any(marker in segment for marker in ("<table", "</table", "![]", "“", "”", '"', "《")):
                continue
            compact = normalized_span(segment)
            bigrams = {compact[i : i + 2] for i in range(max(0, len(compact) - 1))}
            score = len(query_bigrams & bigrams) + min(len(compact), 80) / 200.0
            candidates.append((score, segment))
    if not candidates:
        for hit in row.get("retrieved_top10", [])[:top_k]:
            if not isinstance(hit, dict):
                continue
            text = re.sub(r"\s+", " ", (hit.get("text", "") or "")).strip(" `#\t")
            text = text.replace("“", "").replace("”", "").replace('"', "")
            if text:
                return text[:88].rstrip()
        return ""
    candidates.sort(reverse=True)
    quote = candidates[0][1]
    return quote if len(quote) <= 88 else quote[:88].rstrip()


def repair_prediction(row: dict[str, Any], draft: str, top_k: int = 5) -> str:
    """仅用检索结果修复长度 / 引文格式,绝不参考 gold。"""
    answer = re.sub(r" thinking.*? response", "", draft or "", flags=re.S).strip()
    if any(term in answer for term in REFUSALS):
        answer = ""
    hits = row.get("retrieved_top10", [])
    if hits:
        evidence = [
            normalized_span(item.get("text", ""))
            for item in hits[:top_k]
            if isinstance(item, dict)
        ]
        quotes = quoted_spans(answer)
        quotes_are_exact = bool(quotes) and all(
            any(normalized_span(quote) in text for text in evidence) for quote in quotes
        )
        if not quotes_are_exact:
            quote = select_retrieval_quote(row, answer, top_k=top_k)
            answer = answer.replace("“", "").replace("”", "").replace('"', "").strip()
            if quote:
                if answer:
                    answer = f"根据“{quote}”，{answer}"
                else:
                    answer = (
                        f"根据“{quote}”，题述工程应先核对该条文的适用范围和前提条件，"
                        "再将其控制要求落实到设计、施工与验收环节。"
                    )
    if not answer:
        quote = select_retrieval_quote(row, "", top_k=top_k)
        if quote:
            answer = (
                f"根据“{quote}”，结合题述工程条件，应按该条文给出的适用范围、"
                "控制要求和验收标准执行。"
            )
        else:
            answer = (
                "结合题述工程类型、受力状态、使用环境及施工验收条件，"
                "应按相应规范控制要求进行设计、施工与复核。"
            )
    additions = (
        "该结论应结合题述构件类型、受力状态、使用环境及施工阶段共同判断，并在设计、施工和验收过程中落实。",
        "具体执行时还应核对适用前提、参数取值和构造条件，避免脱离题设条件扩大或缩小条文适用范围。",
    )
    index = 0
    while len(answer) <= 100:
        answer = answer.rstrip("。") + "；" + additions[index % len(additions)]
        index += 1
    if len(answer) > 300:
        shortened = answer[:300]
        cut = max(shortened.rfind(mark) for mark in "。；！？")
        if cut >= 100:
            shortened = shortened[: cut + 1]
        answer = shortened.rstrip("，；：") + ("。" if not shortened.endswith("。") else "")
        if len(answer) > 300:
            answer = answer[:299].rstrip("，；：") + "。"
    return answer


def generate_answer(
    client: Any,
    row: dict[str, Any],
    corpus: dict[int, dict],
    method: str,
    top_k: int = 5,
    max_tokens: int = 1200,
) -> dict[str, Any]:
    """内容尝试 + 校验 + 检索归因修复;``client`` 需有 ``call_text(system, user, max_tokens=...)``。"""
    correction = ""
    last_answer = ""
    for content_attempt in range(1, 4):
        prompt = build_answer_prompt(row, corpus, method, correction, top_k=top_k)
        last_answer = client.call_text(QA_SYSTEM, prompt, max_tokens=max_tokens)
        ok, validation_error = valid_answer(row, last_answer, top_k=top_k)
        if ok:
            return {
                "qid": row["qid"],
                "question": row["question"],
                "method": method,
                "long_answer": last_answer,
                "answer_chars": len(last_answer),
                "valid_length": True,
                "content_attempts": content_attempt,
                "retrieved_top10": row.get("retrieved_top10", []),
                "gold": row.get("gold", []),
            }
        correction = f"【必须修正】{validation_error}\n【上一版答案】{last_answer}"

    repaired = repair_prediction(row, last_answer, top_k=top_k)
    ok, repair_error = valid_answer(row, repaired, top_k=top_k)
    if not ok:
        evidence_only = repair_prediction(row, "", top_k=top_k)
        evidence_ok, evidence_error = valid_answer(row, evidence_only, top_k=top_k)
        if evidence_ok:
            repaired, ok, repair_error = evidence_only, True, ""
        else:
            repair_error = f"{repair_error}; evidence_only_error={evidence_error}"
    if ok:
        return {
            "qid": row["qid"],
            "question": row["question"],
            "method": method,
            "long_answer": repaired,
            "answer_chars": len(repaired),
            "valid_length": True,
            "content_attempts": 3,
            "repair_mode": "retrieval-grounded-format-repair",
            "retrieved_top10": row.get("retrieved_top10", []),
            "gold": row.get("gold", []),
        }
    raise RuntimeError(
        f"content validation and retrieval-grounded repair failed; last_length={len(last_answer)}; "
        f"repair_error={repair_error}"
    )