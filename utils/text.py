"""Text normalization / tokenization / term extraction utilities."""

from __future__ import annotations

import hashlib
import re
import warnings
from typing import List

from config import STOP_EN, STOP_ZH

try:
    with warnings.catch_warnings():
        # jieba imports pkg_resources, which emits a deprecation warning on import.
        warnings.simplefilter("ignore")
        import jieba

    jieba.setLogLevel(60)
    _HAS_JIEBA = True
except Exception:  # pragma: no cover - regex fallback when jieba is unavailable
    _HAS_JIEBA = False


def normalize_space(text) -> str:
    """Collapse whitespace and strip (same as run_construction_review_v2.normalize_space)."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalized_chars(text) -> str:
    """Keep only digits / latin / CJK and lowercase (used by review-pipeline metrics)."""
    return re.sub(r"[^0-9A-Za-z一-鿿]", "", str(text or "")).lower()


def normalized_span(text) -> str:
    """Drop whitespace and the clause-commentary placeholder (for verbatim quote checks)."""
    return re.sub(r"\s+", "", (text or "").replace("▼ 展开条文说明", "").replace("▼", ""))


def quoted_spans(answer) -> List[str]:
    """Extract quoted contiguous source spans."""
    return re.findall(r'["“]([^"”]+)["”]', answer or "")


def safe_name(text) -> str:
    """Sanitize a string for use as a file name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def doc_cache_key(text) -> str:
    """Canonical SHA-1 cache key for a clause / corpus chunk (same as eval_rag_at5)."""
    return hashlib.sha1(normalize_space(text).encode("utf-8")).hexdigest()


def extract_terms(text, max_terms: int = 48) -> List[str]:
    """Query / clause term extraction (same as eval_rag_at5.extract_terms).

    For Chinese, prefer jieba words of 2+ Han characters; for English, proper
    nouns + lowercased words + adjacent bigrams; stop words are removed.
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


# ---- Deterministic graph construction (ported from build_graphrag_novel_context5_graph_local.py) ----

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