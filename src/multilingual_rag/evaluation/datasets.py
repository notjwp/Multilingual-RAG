"""Dataset loading for offline RAG evaluation.

Two shapes coexist:
  * fixture mode — ``EvaluationExample`` rows with precomputed ``retrieved_document_ids``
    (the legacy ``sample_qa.jsonl`` path, still used by tests).
  * live mode — an ``EvalCorpus`` (documents + queries) that the harness runs through the real
    pipeline. Built from the M0 XQuAD corpus in ``data/eval/xquad/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field


class EvaluationExample(BaseModel):
    """One offline evaluation example (fixture mode, or a scored live query).

    Generation-side fields stay unset when no answer was generated (retrieval-only runs, or
    queries outside a generation sample) — that absence is what marks an example as
    retrieval-only when scoring.
    """

    question: str = Field(min_length=1)
    expected_document_ids: tuple[str, ...] = Field(default_factory=tuple)
    retrieved_document_ids: tuple[str, ...] = Field(default_factory=tuple)
    expected_language: str | None = None
    answer_language: str | None = None
    cited_document_ids: tuple[str, ...] = Field(default_factory=tuple)
    faithful: bool | None = None


@dataclass(frozen=True)
class EvalDocument:
    """One corpus document (a gold paragraph or a distractor), indexed as a single chunk."""

    document_id: str
    text: str
    language: str


@dataclass(frozen=True)
class EvalQuery:
    """One query with its known-relevant document ids (its aligned gold paragraph)."""

    question: str
    expected_document_ids: tuple[str, ...]
    language: str


@dataclass(frozen=True)
class EvalCorpus:
    """A live-mode corpus: documents to index and queries to run against them."""

    documents: tuple[EvalDocument, ...] = field(default_factory=tuple)
    queries: tuple[EvalQuery, ...] = field(default_factory=tuple)


def load_jsonl_dataset(path: Path) -> tuple[EvaluationExample, ...]:
    """Load evaluation examples from a JSONL file (fixture mode)."""
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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_xquad_corpus(
    directory: Path,
    languages: tuple[str, ...],
    *,
    sample: int | None = None,
) -> EvalCorpus:
    """Build a live-mode corpus from the M0 XQuAD files (gold / queries / distractors).

    Document ids are deterministic (``xquad-{lang}-gold-{idx}`` / ``xquad-{lang}-dist-{docid}``)
    and each query's relevant document is its aligned gold paragraph. ``sample`` caps distractors
    and queries per language for fast wiring runs — fewer distractors inflate recall, so a
    sampled run is a smoke check, not a real baseline.
    """
    documents: list[EvalDocument] = []
    queries: list[EvalQuery] = []

    for language in languages:
        gold = _read_jsonl(directory / f"gold_{language}.jsonl")
        distractors = _read_jsonl(directory / f"distractors_{language}.jsonl")
        raw_queries = _read_jsonl(directory / f"queries_{language}.jsonl")

        if sample is not None:
            distractors = distractors[:sample]
            raw_queries = raw_queries[:sample]

        for row in gold:
            documents.append(
                EvalDocument(
                    document_id=f"xquad-{language}-gold-{row['para_idx']}",
                    text=str(row["text"]),
                    language=language,
                )
            )
        for row in distractors:
            documents.append(
                EvalDocument(
                    document_id=f"xquad-{language}-dist-{row['docid']}",
                    text=str(row["text"]),
                    language=language,
                )
            )
        for row in raw_queries:
            queries.append(
                EvalQuery(
                    question=str(row["question"]),
                    expected_document_ids=(f"xquad-{language}-gold-{row['para_idx']}",),
                    language=language,
                )
            )

    return EvalCorpus(documents=tuple(documents), queries=tuple(queries))

