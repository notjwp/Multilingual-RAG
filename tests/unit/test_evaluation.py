from pathlib import Path

import pytest

from multilingual_rag.evaluation.datasets import load_jsonl_dataset
from multilingual_rag.evaluation.faithfulness import FaithfulnessJudge, average_faithfulness
from multilingual_rag.evaluation.metrics import (
    citation_precision,
    citation_recall,
    language_match_rate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from multilingual_rag.evaluation.run import build_report


def test_retrieval_metrics() -> None:
    expected = ("doc-a", "doc-b")
    retrieved = ("doc-x", "doc-b", "doc-a")

    assert recall_at_k(expected, retrieved, k=2) == 0.5
    assert reciprocal_rank(expected, retrieved) == 0.5
    assert ndcg_at_k(expected, retrieved, k=3) > 0


def test_retrieval_metrics_reject_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be greater"):
        recall_at_k(("doc-a",), ("doc-a",), k=0)


def test_language_match_rate() -> None:
    assert language_match_rate(("en", "fr", None), ("en", "en", "es")) == 0.5


def test_citation_precision_and_recall() -> None:
    # cited two sources, one of which is relevant; two sources are relevant overall.
    assert citation_precision(("doc-a", "doc-x"), ("doc-a", "doc-b")) == 0.5
    assert citation_recall(("doc-a", "doc-x"), ("doc-a", "doc-b")) == 0.5


def test_citation_precision_is_vacuously_exact_when_nothing_cited() -> None:
    # Documented convention: no citations -> no wrong citations -> precision 1.0,
    # but recall is 0.0, so the pair still penalises a non-citing answer.
    assert citation_precision((), ("doc-a",)) == 1.0
    assert citation_recall((), ("doc-a",)) == 0.0


class _FakeJudge:
    """Judges an answer faithful iff it contains the word 'grounded'."""

    def is_supported(self, *, answer: str, context: str) -> bool:
        del context
        return "grounded" in answer


def test_average_faithfulness_over_a_judge() -> None:
    judge: FaithfulnessJudge = _FakeJudge()
    verdicts = [
        judge.is_supported(answer="a grounded claim", context="c"),
        judge.is_supported(answer="an unsupported claim", context="c"),
    ]
    assert average_faithfulness(verdicts) == 0.5
    assert average_faithfulness([]) == 0.0


def test_load_jsonl_dataset_and_build_report(tmp_path: Path) -> None:
    dataset_path = tmp_path / "eval.jsonl"
    dataset_path.write_text(
        '{"question":"Q?","expected_document_ids":["doc-a"],'
        '"retrieved_document_ids":["doc-a"],"expected_language":"en","answer_language":"en"}\n',
        encoding="utf-8",
    )

    examples = load_jsonl_dataset(dataset_path)
    report = build_report(examples, k=1)

    assert len(examples) == 1
    assert report["example_count"] == 1
    assert report["recall_at_1"] == 1.0
    assert report["language_match_rate"] == 1.0

