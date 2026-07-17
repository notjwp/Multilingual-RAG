"""Harness wiring test with fakes — proves the pipeline plumbing without loading a model.

Real embedding/vector-store are exercised in the free live baseline (opt-in); here we only
verify that documents flow in, the right documents come back out, and metrics are computed.
"""

from collections.abc import Sequence

from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import DocumentChunk, VectorSearchResult
from multilingual_rag.embeddings.base import EmbeddingVector
from multilingual_rag.evaluation.datasets import EvalCorpus, EvalDocument, EvalQuery
from multilingual_rag.evaluation.harness import run_live_evaluation
from multilingual_rag.evaluation.metrics import recall_at_k
from multilingual_rag.vectorstores.base import VectorFilter


class FakeEmbeddingProvider:
    """'alpha' texts embed to [1,0]; everything else to [0,1] — deterministic and separable."""

    def embed_documents(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        return [self._vec(text) for text in texts]

    def embed_query(self, text: str) -> EmbeddingVector:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> EmbeddingVector:
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]


class FakeVectorStore:
    """In-memory store doing exact dot-product search, scoped by user_id."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, DocumentChunk, EmbeddingVector]] = []

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingVector],
        *,
        user_id: str,
    ) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.rows.append((user_id, chunk, embedding))

    def search(
        self,
        query_embedding: EmbeddingVector,
        *,
        user_id: str,
        top_k: int,
        filters: VectorFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        del filters
        scored = [
            (sum(a * b for a, b in zip(query_embedding, vec, strict=True)), chunk)
            for row_user, chunk, vec in self.rows
            if row_user == user_id
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return tuple(
            VectorSearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                language=chunk.language,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=score,
                token_count=chunk.token_count,
            )
            for score, chunk in scored[:top_k]
        )

    def delete_document(self, document_id: str, *, user_id: str) -> None:
        raise NotImplementedError


def test_harness_ingests_retrieves_and_scores() -> None:
    corpus = EvalCorpus(
        documents=(
            EvalDocument(document_id="gold-0", text="alpha document about defense", language="en"),
            EvalDocument(document_id="dist-1", text="beta unrelated passage", language="en"),
            EvalDocument(document_id="dist-2", text="beta another passage", language="en"),
        ),
        queries=(
            EvalQuery(
                question="alpha question?",
                expected_document_ids=("gold-0",),
                language="en",
            ),
        ),
    )

    examples = run_live_evaluation(
        settings=Settings(environment="test"),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
        corpus=corpus,
        top_k=3,
    )

    assert len(examples) == 1
    example = examples[0]
    # The 'alpha' query must retrieve the 'alpha' gold doc first.
    assert example.retrieved_document_ids[0] == "gold-0"
    assert example.expected_document_ids == ("gold-0",)
    assert recall_at_k(example.expected_document_ids, example.retrieved_document_ids, k=1) == 1.0
