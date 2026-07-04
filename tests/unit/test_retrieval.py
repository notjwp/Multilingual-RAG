from collections.abc import Sequence

from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import VectorSearchResult
from multilingual_rag.embeddings.base import EmbeddingVector
from multilingual_rag.retrieval.service import RetrievalService
from multilingual_rag.vectorstores.base import VectorFilter


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> EmbeddingVector:
        self.queries.append(text)
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.search_calls: list[tuple[EmbeddingVector, int, VectorFilter | None]] = []

    def search(
        self,
        query_embedding: EmbeddingVector,
        *,
        top_k: int,
        filters: VectorFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        self.search_calls.append((query_embedding, top_k, filters))
        return (
            VectorSearchResult(
                chunk_id="chunk-1",
                document_id="doc-1",
                text="retrieved text",
                language="en",
                source="sample.txt",
                chunk_index=0,
                score=0.9,
                token_count=2,
            ),
        )

    def upsert_chunks(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError

    def delete_document(self, document_id: str) -> None:
        raise NotImplementedError


def test_retrieval_service_embeds_query_and_searches_store() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    service = RetrievalService(
        Settings(retrieval_top_k=3),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    context = service.retrieve("  What is RAG?  ", filters={"language": "en"})

    assert context.query == "What is RAG?"
    assert embedding_provider.queries == ["What is RAG?"]
    assert vector_store.search_calls == [([1.0, 0.0], 3, {"language": "en"})]
    assert context.results[0].chunk_id == "chunk-1"

