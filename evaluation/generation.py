"""Generation metrics: char / token P-R-F1, EM, ROUGE-L, BLEU1.

Ported from three places in the experiment code, keeping names and conventions:
- ``char_scores``/``token_scores``/``rouge_l_f1``/``bleu1``: GraphRAG-Bench style
  (``scripts/eval_rag_at5.py``).
- ``char_prf``: main clause-QA convention -- whitespace-stripped character
  multiset (``score_and_emit_tables_20260713.py``); normalized to [0,1] here
  (the source multiplies by 100 for display).
- ``token_prf``: SQuAD official normalization (lowercase, strip punctuation, drop a/an/the).
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

_TEXT_RE = re.compile(r"[^0-9a-z一-鿿\s]")


def normalize_answer(text) -> str:
    """GraphRAG-Bench style: lowercase, keep CJK/digits/latin, collapse whitespace."""
    text = str(text or "").lower()
    text = _TEXT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def official_normalize(text) -> str:
    """SQuAD official normalization: lowercase, strip punctuation, drop a/an/the."""
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
    """Main clause-QA convention: whitespace-stripped character multiset F1, normalized to [0,1]."""
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
    """Token F1 after SQuAD official normalization, normalized to [0,1]."""
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