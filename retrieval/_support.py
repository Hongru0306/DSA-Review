"""Shared helpers for the real-RAG adapters: lazy imports and ``[DOCID:i]`` mapping.

Each adapter indexes the corpus with a ``[DOCID:i]`` marker prepended to every
document, then parses those markers back out of the retrieval context. This maps
an arbitrary framework's retrieved context to corpus doc indices without
depending on that framework's internal id scheme.
"""

from __future__ import annotations

import importlib
import re
from typing import Any, Iterable, List

DOCID_RE = re.compile(r"\[DOCID:(\d+)\]")


def require(module_name: str, install_hint: str) -> Any:
    """Import an optional baseline dependency, raising a clear error if missing."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Optional baseline dependency '{module_name}' is not installed. {install_hint}"
        ) from exc


def tag_docs(corpus: Iterable[str]) -> List[str]:
    """Prepend a ``[DOCID:i]`` marker to each corpus doc."""
    return [f"[DOCID:{i}]\n{text}" for i, text in enumerate(corpus)]


def parse_doc_ids(text: Any, corpus_size: int, k: int | None = None) -> List[int]:
    """Extract unique in-range ``[DOCID:i]`` indices in first-seen order."""
    seen: List[int] = []
    for match in DOCID_RE.finditer(str(text or "")):
        index = int(match.group(1))
        if 0 <= index < corpus_size and index not in seen:
            seen.append(index)
            if k is not None and len(seen) >= k:
                break
    return seen