"""Provider-independent generation helpers: prompts, citation parsing, language resolution."""

from multilingual_rag.core.models import RetrievalContext, VectorSearchResult
from multilingual_rag.generation.citations import parse_cited_results
from multilingual_rag.generation.language import resolve_answer_language
from multilingual_rag.generation.prompts import build_answer_prompt


def _results(*chunk_ids: str, language: str = "en") -> tuple[VectorSearchResult, ...]:
    return tuple(
        VectorSearchResult(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            text=f"text {chunk_id}",
            language=language,
            source="sample.txt",
            chunk_index=index,
            score=0.9,
            token_count=2,
        )
        for index, chunk_id in enumerate(chunk_ids)
    )


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


def test_parse_cited_results_maps_markers_in_order() -> None:
    cited = parse_cited_results("Uses [2] then [1].", _results("a", "b", "c"))
    assert [r.chunk_id for r in cited] == ["b", "a"]


def test_parse_cited_results_ignores_out_of_range_markers() -> None:
    cited = parse_cited_results("Claim [9] and [0] and [2].", _results("a", "b"))
    assert [r.chunk_id for r in cited] == ["b"]  # [9]/[0] out of range, [2] -> results[1]


def test_parse_cited_results_dedupes_preserving_first_seen() -> None:
    cited = parse_cited_results("[1] [2] [1] [2]", _results("a", "b"))
    assert [r.chunk_id for r in cited] == ["a", "b"]


def test_parse_cited_results_cites_nothing_without_markers() -> None:
    assert parse_cited_results("A plain answer with no brackets.", _results("a", "b")) == ()


def test_resolve_language_prefers_explicit_preference() -> None:
    assert resolve_answer_language("fr", "en", ()) == "fr"


def test_resolve_language_uses_known_query_language() -> None:
    assert resolve_answer_language(None, "de", ()) == "de"


def test_resolve_language_falls_back_to_evidence_when_query_unknown() -> None:
    # Short query -> "unknown"; answer should follow the retrieved documents, not say "unknown".
    assert resolve_answer_language(None, "unknown", _results("a", language="zh-cn")) == "zh-cn"


def test_resolve_language_defaults_to_en_when_nothing_known() -> None:
    assert resolve_answer_language(None, "unknown", ()) == "en"
