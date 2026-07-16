from pathlib import Path

import pytest

from multilingual_rag.evaluation.datasets import load_jsonl_dataset
from multilingual_rag.evaluation.metrics import (
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

