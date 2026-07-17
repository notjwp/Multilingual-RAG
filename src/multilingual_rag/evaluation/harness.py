"""Run the real retrieval pipeline over an evaluation corpus.

Replaces the old "score a static fixture" approach: this ingests documents through the actual
``VectorStore`` and queries them through the actual ``RetrievalService``, so retrieval quality
is measured rather than assumed. Provider-agnostic — it takes the embedding/vector-store ports,
so tests inject fakes and the real run uses bge-m3 + Chroma for free.

Generation-side metrics (citations, faithfulness, answer language) need an LLM and land in
Phase C; this module measures retrieval only.
"""

from __future__ import annotations

from collections.abc import Sequence

from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import DocumentChunk
from multilingual_rag.embeddings.base import EmbeddingProvider
from multilingual_rag.evaluation.datasets import (
    EvalCorpus,
    EvalDocument,
    EvaluationExample,
)
from multilingual_rag.ingestion.service_utils import checksum_text
from multilingual_rag.retrieval.service import RetrievalService
from multilingual_rag.vectorstores.base import VectorStore

EVAL_USER_ID = "__eval__"


def _chunk_for(document: EvalDocument) -> DocumentChunk:
    """Wrap a corpus document as a single retrievable chunk."""
    return DocumentChunk(
        chunk_id=f"{document.document_id}:0",
        document_id=document.document_id,
        text=document.text,
        language=document.language,
        source=document.document_id,
        chunk_index=0,
        checksum=checksum_text(document.text),
        token_count=len(document.text.split()),
    )


def ingest_documents(
    vector_store: VectorStore,
    embedding_provider: EmbeddingProvider,
    documents: Sequence[EvalDocument],
    *,
    user_id: str = EVAL_USER_ID,
    batch_size: int = 1000,
) -> int:
    """Embed and upsert corpus documents in batches. Returns the number indexed."""
    docs = list(documents)
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        embeddings = embedding_provider.embed_documents(tuple(doc.text for doc in batch))
        vector_store.upsert_chunks(
            tuple(_chunk_for(doc) for doc in batch), embeddings, user_id=user_id
        )
    return len(docs)


def run_live_evaluation(
    *,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    corpus: EvalCorpus,
    top_k: int,
    user_id: str = EVAL_USER_ID,
) -> tuple[EvaluationExample, ...]:
    """Index the corpus, run every query through real retrieval, return scored examples.

    Emits ``EvaluationExample`` so the same ``build_report`` scores live and fixture runs.
    ``answer_language`` is left unset — generation is Phase C.
    """
    ingest_documents(vector_store, embedding_provider, corpus.documents, user_id=user_id)

    # Construct retrieval from the SAME instances used to ingest, so the run is self-consistent.
    retrieval_service = RetrievalService(
        settings,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    examples: list[EvaluationExample] = []
    for query in corpus.queries:
        context = retrieval_service.retrieve(query.question, user_id=user_id, top_k=top_k)
        examples.append(
            EvaluationExample(
                question=query.question,
                expected_document_ids=query.expected_document_ids,
                retrieved_document_ids=tuple(result.document_id for result in context.results),
                expected_language=query.language,
                answer_language=None,
            )
        )
    return tuple(examples)
