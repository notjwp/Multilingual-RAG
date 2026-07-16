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
        self.search_calls: list[tuple[EmbeddingVector, str, int, VectorFilter | None]] = []

    def search(
        self,
        query_embedding: EmbeddingVector,
        *,
        user_id: str,
        top_k: int,
        filters: VectorFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        self.search_calls.append((query_embedding, user_id, top_k, filters))
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

    def delete_document(self, document_id: str, *, user_id: str) -> None:
        raise NotImplementedError


def test_retrieval_service_embeds_query_and_scopes_search_to_user() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    service = RetrievalService(
        Settings(retrieval_top_k=3),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    context = service.retrieve("  What is RAG?  ", user_id="user-1", filters={"language": "en"})

    assert context.query == "What is RAG?"
    assert embedding_provider.queries == ["What is RAG?"]
    # user_id must reach the vector store — this is the tenancy boundary.
    assert vector_store.search_calls == [([1.0, 0.0], "user-1", 3, {"language": "en"})]
    assert context.results[0].chunk_id == "chunk-1"

