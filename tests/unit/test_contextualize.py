"""Condense-prompt helpers for history-aware retrieval (no network/model)."""

from __future__ import annotations

from multilingual_rag.core.models import ConversationTurn
from multilingual_rag.generation.contextualize import (
    build_contextualize_prompt,
    clean_standalone_query,
)


def _history() -> tuple[ConversationTurn, ...]:
    return (
        ConversationTurn(role="user", content="Tell me about the Zorblax Protocol"),
        ConversationTurn(role="assistant", content="It is a fictional networking standard."),
    )


def test_prompt_includes_history_and_follow_up() -> None:
    prompt = build_contextualize_prompt(_history(), "who founded it?")
    assert "Zorblax Protocol" in prompt  # a prior turn is present
    assert "who founded it?" in prompt  # the follow-up is present
    assert prompt.rstrip().endswith("Standalone question:")


def test_clean_strips_surrounding_quotes_and_whitespace() -> None:
    assert clean_standalone_query('  "Who founded it?"  ', fallback="x") == "Who founded it?"


def test_clean_falls_back_when_empty() -> None:
    assert clean_standalone_query("   ", fallback="original question") == "original question"


def test_clean_caps_length() -> None:
    assert len(clean_standalone_query("a" * 1000, fallback="x")) == 400
