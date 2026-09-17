"""Text and IO utilities."""

from utils.io import append_jsonl, read_json, read_jsonl, write_json, write_jsonl
from utils.text import (
    PROPER_RE,
    SENTENCE_RE,
    TOKEN_RE,
    doc_cache_key,
    extract_terms,
    graph_tokenize,
    normalized_chars,
    normalized_entity,
    normalized_span,
    normalize_space,
    quoted_spans,
    safe_name,
)

__all__ = [
    "append_jsonl", "read_json", "read_jsonl", "write_json", "write_jsonl",
    "doc_cache_key", "extract_terms", "normalize_space", "normalized_chars",
    "normalized_span", "normalized_entity", "quoted_spans", "safe_name",
    "graph_tokenize", "TOKEN_RE", "SENTENCE_RE", "PROPER_RE",
]