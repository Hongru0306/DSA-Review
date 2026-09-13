"""条款归因文档审查(论文 Algorithm 3):逐句检索 -> 门控审查 -> 解析打分。

移植 ``scripts/run_construction_review_v2.py`` 的 ``evidence_block`` /
``review_generation_prompt`` / ``parse_has_error`` /
``extract_predicted_correct_sentence`` / ``dual_generation_score_fields``。
"""

from __future__ import annotations

import re
from typing import Any

from evaluation.revision import revision_char_prf, revision_diff_char_prf, revision_embedding_chunks
from generation.prompts import (
    REVIEW_PROMPT_GATE,
    REVIEW_PROMPT_PRE_GATE,
    REVIEW_RETRY_INSTRUCTION,
    REVIEW_SCENARIO_NOTE,
    REVIEW_SCENARIO_NOTE_PRE,
    REVIEW_SYSTEM,
)
from utils.text import normalized_chars, normalize_space

GENERATION_SCORE_VERSION = "unconditional-direct-revision-char-plus-diff-and-embedding-f1-v6"


def direct_clause_text(text: Any, limit: int = 900) -> str:
    """去除 "▼" 后的条文说明,取规范空白后前 limit 字符。"""
    value = str(text or "").split("▼", 1)[0]
    return normalize_space(value)[:limit]


def full_clause_text(text: Any) -> str:
    """去条文说明、不截断的正文。"""
    return normalize_space(str(text or "").split("▼", 1)[0])


def evidence_block(retrieval_row: dict[str, Any], top_k: int = 10) -> str:
    hits = retrieval_row.get("retrieved_top10", [])[:top_k]
    if not hits:
        return "（本方法不提供检索证据。）"
    chunks: list[str] = []
    include_scenario_keywords = retrieval_row.get("method") == "Ours"
    for index, hit in enumerate(hits, 1):
        chunk = (
            f"[{index}] {hit.get('spec_name', hit.get('spec_id', ''))} "
            f"第{hit.get('clause', '')}条\n"
            f"{direct_clause_text(hit.get('text', ''), 900)}"
        )
        if include_scenario_keywords:
            keywords = [
                normalize_space(value)
                for value in hit.get("scenario_keywords", [])
                if normalize_space(value)
            ]
            chunk += "\n本次查询匹配的条文场景词：" + ("、".join(keywords) if keywords else "（无）")
        chunks.append(chunk)
    return "\n\n".join(chunks)


def review_generation_prompt(
    row: dict[str, Any],
    retrieval_row: dict[str, Any],
    use_direct_gate: bool = True,
) -> str:
    """审查 prompt;``use_direct_gate=True`` 用 batch-86 起的"直接冲突门控"版本。"""
    scenario_note = ""
    if retrieval_row.get("method") == "Ours":
        scenario_note = REVIEW_SCENARIO_NOTE if use_direct_gate else REVIEW_SCENARIO_NOTE_PRE
    template = REVIEW_PROMPT_GATE if use_direct_gate else REVIEW_PROMPT_PRE_GATE
    return template.format(
        scenario_note=scenario_note,
        scenario_context=row.get("scenario_context", ""),
        review_sentence=row.get("review_sentence", ""),
        evidence_block=evidence_block(retrieval_row),
    )


def generate_review(
    client: Any,
    row: dict[str, Any],
    retrieval_row: dict[str, Any],
    max_tokens: int = 850,
    temperature: float = 0.0,
    max_attempts: int = 4,
) -> str:
    """调用 LLM 并用格式校验重试;``client`` 需有 ``call_text(system, user, max_tokens=..., temperature=...)``。"""
    prompt = review_generation_prompt(row, retrieval_row)
    prediction = client.call_text(REVIEW_SYSTEM, prompt, max_tokens=max_tokens, temperature=temperature)
    for _ in range(1, max_attempts):
        if parse_has_error(prediction) is not None:
            break
        prediction = client.call_text(
            REVIEW_SYSTEM,
            prompt + "\n" + REVIEW_RETRY_INSTRUCTION,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return prediction


# ---- 解析 ----

def parse_has_error(text: Any) -> bool | None:
    value = normalize_space(text)
    if re.search(r"修改建议[：:]\s*(?:无错误|无误|符合|无需修改)", value):
        return False
    if re.search(r"修改建议[：:].*(?:有错误|有误|不符合|应改为|建议改为)", value):
        return True
    if "无错误，无需修改" in value:
        return False
    if any(token in value for token in ("有错误", "建议改为", "应改为", "不符合")):
        return True
    return None


def extract_predicted_correct_sentence(prediction: Any, review_sentence: Any) -> str:
    """从固定两段式答案中提取最终替换句;合规当且仅当保持原句。"""
    text = str(prediction or "").strip()
    matches = list(re.finditer(r"修改建议[：:]", text))
    if not matches:
        return ""
    section = text[matches[-1].end() :].strip()
    if re.search(r"无错误|无误|无需修改", section):
        return normalize_space(review_sentence)
    quoted = re.search(
        r"(?:建议|应)?改为[：:]\s*[“\"'‘](.*?)[”\"'’]\s*[。.]?\s*$",
        section,
        flags=re.DOTALL,
    )
    if quoted:
        return normalize_space(quoted.group(1)).strip("。.")
    plain = re.search(r"(?:建议|应)?改为[：:]\s*(.+?)\s*$", section, flags=re.DOTALL)
    if plain:
        return normalize_space(plain.group(1)).strip("“”\"'‘’。.")
    return ""


def dual_generation_score_fields(prediction: Any, case: dict[str, Any]) -> dict[str, Any]:
    """把单案生成输出解析并打分为各分数域(不含嵌入域,嵌入另行补算)。"""
    prediction_text = str(prediction or "")
    review_sentence = normalize_space(case.get("review_sentence", ""))
    predicted_correct_sentence = extract_predicted_correct_sentence(prediction_text, review_sentence)
    (
        diff_precision,
        diff_recall,
        diff_f1,
        predicted_edit_fragments,
        reference_edit_fragments,
    ) = revision_diff_char_prf(
        review_sentence,
        predicted_correct_sentence,
        case.get("correct_sentence", ""),
    )
    revision_precision, revision_recall, revision_f1 = revision_char_prf(
        predicted_correct_sentence,
        case.get("correct_sentence", ""),
    )
    compliant_correct = (
        not bool(case.get("is_error"))
        and parse_has_error(prediction_text) is False
        and normalized_chars(predicted_correct_sentence)
        == normalized_chars(review_sentence)
    )
    if not bool(case.get("is_error")):
        compliant_score = 1.0 if compliant_correct else 0.0
        diff_precision = compliant_score
        diff_recall = compliant_score
        diff_f1 = compliant_score
    full_precision, full_recall, full_f1 = revision_char_prf(
        prediction_text,
        case.get("gold_answer", ""),
    )
    return {
        "predicted_correct_sentence": predicted_correct_sentence,
        "reference_correct_sentence": normalize_space(case.get("correct_sentence", "")),
        "revision_parse_ok": bool(predicted_correct_sentence),
        "revision_compliant_exact": compliant_correct,
        "predicted_edit_fragments": predicted_edit_fragments,
        "reference_edit_fragments": reference_edit_fragments,
        "revision_diff_char_precision": diff_precision,
        "revision_diff_char_recall": diff_recall,
        "revision_diff_char_f1": diff_f1,
        "revision_embedding_prediction_chunks": revision_embedding_chunks(predicted_correct_sentence),
        "revision_embedding_reference_chunks": revision_embedding_chunks(case.get("correct_sentence", "")),
        "revision_char_precision": revision_precision,
        "revision_char_recall": revision_recall,
        "revision_char_f1": revision_f1,
        "full_answer_char_precision": full_precision,
        "full_answer_char_recall": full_recall,
        "full_answer_char_f1": full_f1,
        "char_precision": full_precision,
        "char_recall": full_recall,
        "char_f1": full_f1,
        "generation_score_version": GENERATION_SCORE_VERSION,
    }