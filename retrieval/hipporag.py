"""HippoRAG baseline adapter (OSU-NLP-Group/HippoRAG).

Wraps ``hipporag.HippoRAG`` behind our ``Retriever`` interface: the corpus is
indexed in ``build`` and ``rank`` retrieves documents and maps them back to
corpus indices by exact text match (HippoRAG returns the original document
strings in ``QuerySolution.docs``).

Requires the optional dependency ``hipporag`` (no PyPI release; install from git)
plus an LLM and embedding endpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from retrieval.base import Retriever
from retrieval._support import require

_INSTALL_HINT = (
    "Install it with `pip install git+https://github.com/OSU-NLP-Group/HippoRAG.git` "
    "(see requirements-baselines.txt)."
)


class HippoRAGRetriever(Retriever):
    name = "HippoRAG"

    def __init__(
        self,
        save_dir: str,
        llm_model_name: str,
        llm_base_url: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        top_k: int = 10,
    ):
        self.save_dir = Path(save_dir)
        self.llm_model_name = llm_model_name
        self.llm_base_url = llm_base_url
        self.embedding_model_name = embedding_model_name
        self.embedding_base_url = embedding_base_url
        self.top_k = top_k
        self.n_docs = 0
        self._docs: List[str] = []
        self._doc_index: dict[str, int] = {}
        self._rag = None

    def build(self, corpus: List[str], encoder) -> None:
        hipporag = require("hipporag", _INSTALL_HINT)
        self._docs = list(corpus)
        self._doc_index = {text: i for i, text in enumerate(self._docs)}
        self.n_docs = len(self._docs)
        self._rag = hipporag.HippoRAG(
            save_dir=str(self.save_dir),
            llm_model_name=self.llm_model_name,
            llm_base_url=self.llm_base_url,
            embedding_model_name=self.embedding_model_name,
            embedding_base_url=self.embedding_base_url,
        )
        self._rag.index(self._docs)

    def rank(self, q: str) -> List[int]:
        require("hipporag", _INSTALL_HINT)
        solutions = self._rag.retrieve([q], num_to_retrieve=self.top_k)
        docs = getattr(solutions[0], "docs", solutions[0])
        ids = [self._doc_index[doc] for doc in docs if doc in self._doc_index]
        return ids if ids else list(range(self.n_docs))