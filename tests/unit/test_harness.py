"""Harness wiring test with fakes — proves the pipeline plumbing without loading a model.

Real embedding/vector-store are exercised in the free live baseline (opt-in); here we only
verify that documents flow in, the right documents come back out, and metrics are computed.
"""

from collections.abc import Sequence

from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import (
    AnswerCitation,
    DocumentChunk,
    GeneratedAnswer,
    RetrievalContext,
    VectorSearchResult,
)
from multilingual_rag.embeddings.base import EmbeddingVector
from multilingual_rag.evaluation.datasets import (
    EvalCorpus,
    EvalDocument,
    EvalQuery,
    EvaluationExample,
)
from multilingual_rag.evaluation.harness import run_live_evaluation
from multilingual_rag.evaluation.metrics import citation_precision, citation_recall, recall_at_k
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
        session_id: str | None = None,
    ) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.rows.append((user_id, chunk, embedding))

    def search(
        self,
        query_embedding: EmbeddingVector,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int,
        filters: VectorFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        del filters, session_id
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

    def delete_document(
        self, document_id: str, *, user_id: str, session_id: str | None = None
    ) -> None:
        raise NotImplementedError


class FakeGenerator:
    """Cites [1] (the top hit) and answers in the evidence's language."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_answer(
        self,
        *,
        context: RetrievalContext,
        preferred_language: str | None = None,
    ) -> GeneratedAnswer:
        del preferred_language
        self.calls += 1
        top = context.results[0]
        return GeneratedAnswer(
            answer="Grounded answer [1]",
            language=top.language,
            citations=(
                AnswerCitation(
                    chunk_id=top.chunk_id,
                    document_id=top.document_id,
                    source=top.source,
                    page=None,
                    text=top.text,
                ),
            ),
        )


class FakeJudge:
    def is_supported(self, *, answer: str, context: str) -> bool:
        del context
        return "Grounded" in answer


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
    # No generator -> generation fields stay unset (marks it retrieval-only when scoring).
    assert example.answer_language is None
    assert example.cited_document_ids == ()
    assert example.faithful is None


def _corpus(query_count: int = 1) -> EvalCorpus:
    return EvalCorpus(
        documents=(
            EvalDocument(document_id="gold-0", text="alpha document about defense", language="en"),
            EvalDocument(document_id="dist-1", text="beta unrelated passage", language="en"),
        ),
        queries=tuple(
            EvalQuery(
                question=f"alpha question {index}?",
                expected_document_ids=("gold-0",),
                language="en",
            )
            for index in range(query_count)
        ),
    )


def _run(**kwargs: object) -> tuple[EvaluationExample, ...]:
    return run_live_evaluation(
        settings=Settings(environment="test"),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
        top_k=2,
        **kwargs,  # type: ignore[arg-type]
    )


def test_harness_scores_generation_when_a_generator_is_given() -> None:
    examples = _run(corpus=_corpus(), answer_generator=FakeGenerator(), judge=FakeJudge())

    example = examples[0]
    assert example.answer_language == "en"
    assert example.cited_document_ids == ("gold-0",)  # cited the gold doc
    assert example.faithful is True
    assert citation_precision(example.cited_document_ids, example.expected_document_ids) == 1.0
    assert citation_recall(example.cited_document_ids, example.expected_document_ids) == 1.0


def test_generation_is_sampled_so_free_tier_quota_survives() -> None:
    generator = FakeGenerator()
    examples = _run(corpus=_corpus(query_count=5), answer_generator=generator, generate_limit=2)

    # Retrieval covers every query; generation only the sample.
    assert len(examples) == 5
    assert generator.calls == 2
    assert [e.answer_language is not None for e in examples] == [True, True, False, False, False]
