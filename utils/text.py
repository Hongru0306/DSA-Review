"""文本规范化 / 分词 / 抽取工具。"""

from __future__ import annotations

import hashlib
import re
from typing import List

from config import STOP_EN, STOP_ZH

try:
    import jieba

    jieba.setLogLevel(60)
    _HAS_JIEBA = True
except Exception:  # pragma: no cover - 无 jieba 时走正则兜底
    _HAS_JIEBA = False


def normalize_space(text) -> str:
    """合并空白并去除首尾(与 run_construction_review_v2.normalize_space 一致)。"""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalized_chars(text) -> str:
    """仅保留数字 / 英文 / 汉字并小写(审查管线指标用)。"""
    return re.sub(r"[^0-9A-Za-z一-鿿]", "", str(text or "")).lower()


def normalized_span(text) -> str:
    """去空白并移除条文说明占位符(生成引文 verbatim 校验用)。"""
    return re.sub(r"\s+", "", (text or "").replace("▼ 展开条文说明", "").replace("▼", ""))


def quoted_spans(answer) -> List[str]:
    """提取引号包裹的连续原文片段。"""
    return re.findall(r'["“]([^"”]+)["”]', answer or "")


def safe_name(text) -> str:
    """文件名安全化。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def doc_cache_key(text) -> str:
    """条款 / 语料切片的标准 SHA-1 缓存键(与 eval_rag_at5.doc_cache_key 一致)。"""
    return hashlib.sha1(normalize_space(text).encode("utf-8")).hexdigest()


def extract_terms(text, max_terms: int = 48) -> List[str]:
    """查询 / 条款词项抽取(与 eval_rag_at5.extract_terms 一致)。

    中文优先 jieba 二字以上汉字词,英文专名 + 小写词 + 相邻二元组,去停用词。
    """
    text = normalize_space(text)
    terms: List[str] = []
    if _HAS_JIEBA and re.search(r"[一-鿿]", text):
        for w in jieba.cut(text):
            w = w.strip()
            if len(w) >= 2 and re.fullmatch(r"[一-鿿]+", w) and w not in STOP_ZH:
                terms.append(w)
    else:
        terms.extend(re.findall(r"[一-鿿]{2,10}", text))
    for m in re.finditer(r"[A-Z][A-Za-z0-9'’./-]+(?:\s+[A-Z][A-Za-z0-9'’./-]+){0,5}", text):
        terms.append(normalize_space(m.group(0)))
    words = [
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z'’./-]{2,}", text)
        if w.lower() not in STOP_EN
    ]
    terms.extend(words)
    for i in range(len(words) - 1):
        terms.append(f"{words[i]} {words[i + 1]}")
    out, seen = [], set()
    for t in terms:
        if t in STOP_ZH:
            continue
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
        if len(out) >= max_terms:
            break
    return out


# ---- 确定性构图用(移植 build_graphrag_novel_context5_graph_local.py)----

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’.-]{2,}")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
PROPER_RE = re.compile(
    r"\b[A-Z][A-Za-z'’.-]+"
    r"(?:\s+(?:(?:of|the|and|de|la|von|van|del|da|di)|[A-Z][A-Za-z'’.-]+)){0,5}"
)
GRAPH_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "as", "at", "is", "are", "was", "were", "be", "been",
    "being", "that", "this", "these", "those", "which", "who", "what", "when",
    "where", "why", "how", "does", "did", "do", "can", "could", "would",
    "should", "has", "have", "had", "it", "its", "their", "than", "then",
    "also", "not", "only", "using", "use", "into", "out", "upon", "over",
    "under", "about", "after", "before", "through", "between", "among", "but",
    "if", "so", "such", "there", "here", "they", "them", "he", "she", "his",
    "her", "we", "our", "you", "your", "i", "me", "my", "all", "any", "no",
    "one", "two", "first", "second", "said", "say", "says", "may", "might",
    "must", "will", "shall", "very", "more", "most", "much", "many", "some",
    "each", "other", "another", "same", "own", "now", "still", "even", "ever",
}
GENERIC_CAPITALIZED = {
    "chapter", "book", "part", "section", "note", "notes", "illustration",
    "contents", "preface", "introduction", "appendix", "transcriber",
    "produced", "project", "volume", "page", "day", "end", "copyright",
}


def graph_tokenize(text) -> List[str]:
    return [m.group(0).lower().strip(".'’-") for m in TOKEN_RE.finditer(text)]


def normalized_entity(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .,'’\"-_").lower()