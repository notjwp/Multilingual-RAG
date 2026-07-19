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


class FakeTransliterator:
    """Returns a preprogrammed native form; records what it was asked to transliterate."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping
        self.calls: list[tuple[str, str]] = []

    def transliterate(self, text: str, *, target_language: str) -> str:
        self.calls.append((text, target_language))
        return self.mapping.get(text, text)


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
    # No transliterator injected -> single-query path, no transliteration reported.
    assert context.transliteration_applied is False
    assert context.transliterated_query is None


def test_romanized_hindi_query_searches_the_transliterated_form() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    transliterator = FakeTransliterator({"bharat ki rajdhani kya hai": "भारत की राजधानी क्या है"})
    service = RetrievalService(
        Settings(),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        transliterator=transliterator,
    )

    context = service.retrieve("bharat ki rajdhani kya hai", user_id="user-1")

    # Detected as romanized Hindi -> transliterated, and the native form is what gets embedded.
    assert transliterator.calls == [("bharat ki rajdhani kya hai", "hi")]
    assert embedding_provider.queries == ["भारत की राजधानी क्या है"]  # single search, native form
    assert context.transliteration_applied is True
    assert context.transliterated_query == "भारत की राजधानी क्या है"
    assert context.query == "bharat ki rajdhani kya hai"  # original preserved for display


def test_english_query_is_not_transliterated() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    # A real English query has no Hindi markers -> never transliterated (stays same-language).
    transliterator = FakeTransliterator({"what is the capital of france": "translated"})
    service = RetrievalService(
        Settings(),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        transliterator=transliterator,
    )

    context = service.retrieve("what is the capital of france", user_id="user-1")

    assert transliterator.calls == []  # detection rejected it before transliterating
    assert embedding_provider.queries == ["what is the capital of france"]
    assert context.transliteration_applied is False


def test_native_script_query_skips_transliteration() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    transliterator = FakeTransliterator({})  # should never be consulted
    service = RetrievalService(
        Settings(),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        transliterator=transliterator,
    )

    context = service.retrieve("भारत की राजधानी क्या है", user_id="user-1")

    assert transliterator.calls == []  # native query is not Latin-script
    assert embedding_provider.queries == ["भारत की राजधानी क्या है"]  # single embed
    assert context.transliteration_applied is False

