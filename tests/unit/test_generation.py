import pytest
from openai import OpenAIError
from pydantic import SecretStr

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import RetrievalContext, VectorSearchResult
from multilingual_rag.generation.openai_generator import OpenAIAnswerGenerator
from multilingual_rag.generation.prompts import build_answer_prompt


class FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class FakeResponsesClient:
    def __init__(self, *, output_text: str = "Answer [1]", fail: bool = False) -> None:
        self.output_text = output_text
        self.fail = fail
        self.requests: list[tuple[str, str, str]] = []

    def create_response(self, *, model: str, instructions: str, prompt: str) -> FakeResponse:
        self.requests.append((model, instructions, prompt))
        if self.fail:
            raise OpenAIError("generation failed")
        return FakeResponse(self.output_text)


def make_context() -> RetrievalContext:
    return RetrievalContext(
        query="What is this document about?",
        query_language="en",
        results=(
            VectorSearchResult(
                chunk_id="chunk-1",
                document_id="doc-1",
                text="This document explains RAG.",
                language="en",
                source="sample.txt",
                chunk_index=0,
                score=0.9,
                token_count=5,
            ),
        ),
    )


def test_prompt_contains_language_question_and_context() -> None:
    prompt = build_answer_prompt(make_context(), response_language="fr")

    assert "Answer language: fr" in prompt
    assert "What is this document about?" in prompt
    assert "This document explains RAG." in prompt


def test_openai_generator_returns_answer_and_citations() -> None:
    client = FakeResponsesClient(output_text="RAG is explained in the document. [1]")
    generator = OpenAIAnswerGenerator(
        Settings(openai_api_key=SecretStr("test-key"), openai_generation_model="gpt-test"),
        client=client,
    )

    answer = generator.generate_answer(context=make_context(), preferred_language="es")

    assert answer.answer == "RAG is explained in the document. [1]"
    assert answer.language == "es"
    assert answer.citations[0].chunk_id == "chunk-1"
    assert client.requests[0][0] == "gpt-test"


def test_openai_generator_uses_query_language_by_default() -> None:
    generator = OpenAIAnswerGenerator(
        Settings(openai_api_key=SecretStr("test-key")),
        client=FakeResponsesClient(),
    )

    answer = generator.generate_answer(context=make_context())

    assert answer.language == "en"


def test_openai_generator_requires_key_without_injected_client() -> None:
    with pytest.raises(AppError, match="OPENAI_API_KEY"):
        OpenAIAnswerGenerator(Settings(openai_api_key=None))


def test_openai_generation_errors_are_wrapped() -> None:
    generator = OpenAIAnswerGenerator(
        Settings(openai_api_key=SecretStr("test-key")),
        client=FakeResponsesClient(fail=True),
    )

    with pytest.raises(AppError, match="answer generation failed"):
        generator.generate_answer(context=make_context())

