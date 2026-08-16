"""Provider-independent generation helpers: prompts, citation parsing, language resolution."""

from multilingual_rag.core.models import RetrievalContext, VectorSearchResult
from multilingual_rag.generation.citations import (
    parse_cited_results,
    strip_unresolvable_markers,
)
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


# --- dangling citation markers ------------------------------------------------------------------


def test_a_marker_with_no_matching_result_is_removed_from_the_answer() -> None:
    """Seen live: with one resolvable passage the model wrote "… swasthya hi dhan hai. [2]" and
    the UI rendered a superscript citation pointing at nothing."""
    results = _results("c1")

    cleaned = strip_unresolvable_markers("Nahi hai kyonki swasthya hi dhan hai. [2]", results)

    assert cleaned == "Nahi hai kyonki swasthya hi dhan hai."


def test_resolvable_markers_survive_untouched() -> None:
    results = _results("c1", "c2")

    assert strip_unresolvable_markers("Alpha [1] and beta [2].", results) == (
        "Alpha [1] and beta [2]."
    )


def test_only_the_unresolvable_marker_is_dropped() -> None:
    results = _results("c1")

    assert strip_unresolvable_markers("Dekhiye [1] aur [5] mein.", results) == (
        "Dekhiye [1] aur mein."
    )


def test_markers_are_never_renumbered() -> None:
    # Rewriting [5] -> [2] would invent an attribution the model never made.
    results = _results("c1", "c2")

    cleaned = strip_unresolvable_markers("See [2] and [9].", results)

    assert "[2]" in cleaned
    assert "[1]" not in cleaned  # [9] was deleted, not remapped onto a real source


def test_a_refusal_with_no_results_loses_every_marker() -> None:
    assert strip_unresolvable_markers("Your documents don't cover this. [1]", ()) == (
        "Your documents don't cover this."
    )


def test_markers_inside_code_are_left_alone() -> None:
    # Must match the frontend's rehypeCitations, which skips code/pre — if the two disagree the
    # rendered output stops matching the parse.
    results = _results("c1")

    cleaned = strip_unresolvable_markers("Use `arr[2]` here [7].", results)

    assert "`arr[2]`" in cleaned


def test_stripping_does_not_change_which_results_are_cited() -> None:
    results = _results("c1", "c2")
    answer = "Alpha [1] beta [9]."

    cleaned = strip_unresolvable_markers(answer, results)

    assert parse_cited_results(answer, results) == parse_cited_results(cleaned, results)
