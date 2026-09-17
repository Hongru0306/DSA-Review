"""Unified retrieval protocol (the experiment code's duck-typing protocol, made explicit).

Retrieval row schema: ``{"method", "qid"/"question", "topK": [doc_idx, ...],
"metrics": {...}}``. ``rank(q)`` returns the full-corpus ordering ``list[int]``
(best-first); callers slice ``[:top_k]`` as needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class Retriever(ABC):
    """Interface implemented by every retriever."""

    name = "retriever"

    @abstractmethod
    def build(self, corpus: List[str], encoder) -> None:
        """Build the index (corpus is a list of clause / chunk texts, encoder provides embeddings)."""

    @abstractmethod
    def rank(self, q: str) -> List[int]:
        """Return all corpus doc indices sorted by descending relevance."""