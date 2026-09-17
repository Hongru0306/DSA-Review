"""RAPTOR baseline adapter (parthsarthi03/raptor).

Wraps ``raptor.RetrievalAugmentation`` behind our ``Retriever`` interface. The
corpus (each doc tagged with ``[DOCID:i]``) is added as one text in ``build``;
``rank`` retrieves tree context and parses the ``[DOCID:i]`` markers back to
corpus indices.

RAPTOR expects three pluggable models (summarization, QA, embedding). When an
``llm_client`` (with ``call_text``) is supplied, this adapter builds the
summarization / QA models on it; the embedding model wraps the given encoder.

Requires the optional dependency ``raptor``. The upstream repository has no
packaging metadata, so it cannot be pip-installed -- clone it and put its root on
``PYTHONPATH`` (see requirements-baselines.txt).
"""

from __future__ import annotations

from typing import Any, List, Optional

from retrieval.base import Retriever
from retrieval._support import parse_doc_ids, require, tag_docs

_INSTALL_HINT = (
    "RAPTOR is not packaged; clone it and add it to PYTHONPATH:\n"
    "    git clone --depth 1 https://github.com/parthsarthi03/raptor.git third_party/raptor\n"
    "    export PYTHONPATH=$PYTHONPATH:$PWD/third_party/raptor   (Windows: set PYTHONPATH=%PYTHONPATH%;%CD%\\third_party\\raptor)"
)


def _build_default_models(raptor: Any, llm_client: Any, encoder: Any):
    """Construct RAPTOR summarization / QA / embedding models backed by our client and encoder."""
    from raptor import BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel

    class _SummarizationModel(BaseSummarizationModel):
        def summarize(self, context, max_tokens=150):
            text = context if isinstance(context, str) else "\n".join(map(str, context))
            return llm_client.call_text(
                "Summarize the context concisely.", text, max_tokens=max_tokens
            )

    class _QAModel(BaseQAModel):
        def answer_question(self, context, question):
            return llm_client.call_text(
                "Answer strictly from the context.",
                f"Question: {question}\n\nContext:\n{context}",
                max_tokens=200,
            )

    class _EmbeddingModel(BaseEmbeddingModel):
        def create_embedding(self, text):
            return encoder.encode_batch([text])[0]

    return _SummarizationModel(), _QAModel(), _EmbeddingModel()


class RaptorRetriever(Retriever):
    name = "RAPTOR"

    def __init__(
        self,
        llm_client: Any = None,
        summarization_model: Any = None,
        qa_model: Any = None,
        embedding_model: Any = None,
        top_k: int = 10,
        num_layers: int = 5,
    ):
        self.llm_client = llm_client
        self.summarization_model = summarization_model
        self.qa_model = qa_model
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.num_layers = num_layers
        self.n_docs = 0
        self._ra = None

    def build(self, corpus: List[str], encoder) -> None:
        raptor = require("raptor", _INSTALL_HINT)
        summarization_model = self.summarization_model
        qa_model = self.qa_model
        embedding_model = self.embedding_model
        if summarization_model is None or qa_model is None or embedding_model is None:
            if self.llm_client is None:
                raise ValueError(
                    "RaptorRetriever requires either prebuilt models or an llm_client "
                    "to build the summarization / QA / embedding models."
                )
            default_summarization, default_qa, default_embedding = _build_default_models(
                raptor, self.llm_client, encoder
            )
            summarization_model = summarization_model or default_summarization
            qa_model = qa_model or default_qa
            embedding_model = embedding_model or default_embedding
        config = raptor.RetrievalAugmentationConfig(
            summarization_model=summarization_model,
            qa_model=qa_model,
            embedding_model=embedding_model,
            tb_num_layers=self.num_layers,
        )
        self._ra = raptor.RetrievalAugmentation(config=config)
        self.n_docs = len(corpus)
        self._ra.add_documents("\n".join(tag_docs(corpus)))

    def rank(self, q: str) -> List[int]:
        require("raptor", _INSTALL_HINT)
        context, _layer_information = self._ra.retrieve(q, top_k=self.top_k)
        text = context if isinstance(context, str) else "\n".join(map(str, context))
        ids = parse_doc_ids(text, self.n_docs)
        return ids if ids else list(range(self.n_docs))