"""生成答案指标:字符 / 词元 P-R-F1、EM、ROUGE-L、BLEU1。

移植自三处实验代码,名称与口径尽量保留:
- ``char_scores``/``token_scores``/``rouge_l_f1``/``bleu1``: GraphRAG-Bench 风格
  (``scripts/eval_rag_at5.py``)。
- ``char_prf``: 条款问答主口径 —— 去空白后字符多重集(``score_and_emit_tables_20260713.py``),
  输出已归一化到 [0,1](源码再乘 100 展示)。
- ``token_prf``: SQuAD 官方归一化(小写、去标点、去 a/an/the)。
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

_TEXT_RE = re.compile(r"[^0-9a-z一-鿿\s]")


def normalize_answer(text) -> str:
    """GraphRAG-Bench 风格:小写,保留汉字/数字/英文,空白合并。"""
    text = str(text or "").lower()
    text = _TEXT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def official_normalize(text) -> str:
    """SQuAD 官方归一化:小写、去标点、去 a/an/the。"""
    text = (text or "").lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def char_scores(pred: str, gold: str) -> tuple[float, float, float]:
    p = [c for c in str(pred or "") if not c.isspace()]
    g = [c for c in str(gold or "") if not c.isspace()]
    if not p or not g:
        return 0.0, 0.0, 0.0
    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    prec, rec = overlap / len(p), overlap / len(g)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def token_scores(pred: str, gold: str) -> tuple[float, float, float]:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return 0.0, 0.0, 0.0
    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    prec, rec = overlap / len(p), overlap / len(g)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def char_prf(pred, ref) -> tuple[float, float, float]:
    """条款问答主口径:去空白后字符多重集 F1,归一化到 [0,1]。"""
    p = re.sub(r"\s+", "", str(pred or ""))
    g = re.sub(r"\s+", "", str(ref or ""))
    if not p or not g:
        return 0.0, 0.0, 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    precision = overlap / len(p)
    recall = overlap / len(g)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def token_prf(pred, ref) -> tuple[float, float, float]:
    """SQuAD 官方归一化后词元 F1,归一化到 [0,1]。"""
    p = official_normalize(pred).split()
    g = official_normalize(ref).split()
    if not p or not g:
        return 0.0, 0.0, 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    precision = overlap / len(p)
    recall = overlap / len(g)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def exact_match(pred, gold) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def rouge_l_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return 0.0
    dp = [0] * (len(g) + 1)
    for x in p:
        prev = 0
        for j, y in enumerate(g, 1):
            tmp = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = tmp
    lcs = dp[-1]
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def bleu1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    bp = 1.0 if len(p) > len(g) else math.exp(1 - len(g) / max(len(p), 1))
    return bp * overlap / len(p)