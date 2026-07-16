"""Dataset loading for offline RAG evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvaluationExample(BaseModel):
    """One offline evaluation example."""

    question: str = Field(min_length=1)
    expected_document_ids: tuple[str, ...] = Field(default_factory=tuple)
    retrieved_document_ids: tuple[str, ...] = Field(default_factory=tuple)
    expected_language: str | None = None
    answer_language: str | None = None


def load_jsonl_dataset(path: Path) -> tuple[EvaluationExample, ...]:
    """Load evaluation examples from a JSONL file."""
    examples: list[EvaluationExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
        examples.append(EvaluationExample.model_validate(raw))
    return tuple(examples)

