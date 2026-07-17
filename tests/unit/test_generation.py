import pytest
from openai import OpenAIError
from pydantic import SecretStr

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import RetrievalContext, VectorSearchResult
from multilingual_rag.generation.citations import parse_cited_results
from multilingual_rag.generation.language import resolve_answer_language
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


def _results(*chunk_ids: str) -> tuple[VectorSearchResult, ...]:
    return tuple(
        VectorSearchResult(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            text=f"text {chunk_id}",
            language="en",
            source="sample.txt",
            chunk_index=index,
            score=0.9,
            token_count=2,
        )
        for index, chunk_id in enumerate(chunk_ids)
    )


def test_parse_cited_results_maps_markers_in_order() -> None:
    results = _results("a", "b", "c")
    cited = parse_cited_results("Uses [2] then [1].", results)
    assert [r.chunk_id for r in cited] == ["b", "a"]


def test_parse_cited_results_ignores_out_of_range_markers() -> None:
    results = _results("a", "b")
    cited = parse_cited_results("Claim [9] and [0] and [2].", results)
    assert [r.chunk_id for r in cited] == ["b"]  # [9]/[0] out of range, [2] -> results[1]


def test_parse_cited_results_dedupes_preserving_first_seen() -> None:
    results = _results("a", "b")
    cited = parse_cited_results("[1] [2] [1] [2]", results)
    assert [r.chunk_id for r in cited] == ["a", "b"]


def test_parse_cited_results_cites_nothing_without_markers() -> None:
    assert parse_cited_results("A plain answer with no brackets.", _results("a", "b")) == ()


def test_generator_cites_only_marked_chunks_not_all() -> None:
    # Three chunks retrieved, answer cites only [2]; the other two must NOT be cited.
    context = RetrievalContext(query="q", query_language="en", results=_results("a", "b", "c"))
    generator = OpenAIAnswerGenerator(
        Settings(openai_api_key=SecretStr("test-key")),
        client=FakeResponsesClient(output_text="Grounded claim [2]."),
    )

    answer = generator.generate_answer(context=context)

    assert [c.chunk_id for c in answer.citations] == ["b"]


def _zh_results() -> tuple[VectorSearchResult, ...]:
    return (
        VectorSearchResult(
            chunk_id="c1",
            document_id="d1",
            text="内容",
            language="zh-cn",
            source="s",
            chunk_index=0,
            score=0.9,
            token_count=2,
        ),
    )


def test_resolve_language_prefers_explicit_preference() -> None:
    assert resolve_answer_language("fr", "en", ()) == "fr"


def test_resolve_language_uses_known_query_language() -> None:
    assert resolve_answer_language(None, "de", ()) == "de"


def test_resolve_language_falls_back_to_evidence_when_query_unknown() -> None:
    # Short query -> "unknown"; answer should follow the retrieved documents, not say "unknown".
    assert resolve_answer_language(None, "unknown", _zh_results()) == "zh-cn"


def test_resolve_language_defaults_to_en_when_nothing_known() -> None:
    assert resolve_answer_language(None, "unknown", ()) == "en"


def test_generator_never_answers_in_unknown() -> None:
    context = RetrievalContext(query="GDPR?", query_language="unknown", results=_zh_results())
    generator = OpenAIAnswerGenerator(
        Settings(openai_api_key=SecretStr("test-key")),
        client=FakeResponsesClient(output_text="内容 [1]"),
    )

    answer = generator.generate_answer(context=context)

    assert answer.language == "zh-cn"  # resolved from evidence, never "unknown"

