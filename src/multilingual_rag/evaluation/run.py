"""Offline evaluation report CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from multilingual_rag.evaluation.datasets import EvaluationExample, load_jsonl_dataset
from multilingual_rag.evaluation.metrics import (
    language_match_rate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def build_report(examples: tuple[EvaluationExample, ...], *, k: int) -> dict[str, Any]:
    """Build an aggregate evaluation report."""
    recalls = [
        recall_at_k(example.expected_document_ids, example.retrieved_document_ids, k=k)
        for example in examples
    ]
    reciprocal_ranks = [
        reciprocal_rank(example.expected_document_ids, example.retrieved_document_ids)
        for example in examples
    ]
    ndcgs = [
        ndcg_at_k(example.expected_document_ids, example.retrieved_document_ids, k=k)
        for example in examples
    ]
    return {
        "example_count": len(examples),
        f"recall_at_{k}": mean_or_zero(recalls),
        "mrr": mean_or_zero(reciprocal_ranks),
        f"ndcg_at_{k}": mean_or_zero(ndcgs),
        "language_match_rate": language_match_rate(
            tuple(example.expected_language for example in examples),
            tuple(example.answer_language for example in examples),
        ),
    }


def mean_or_zero(values: list[float]) -> float:
    """Return the arithmetic mean or zero for empty inputs."""
    return mean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline RAG evaluation metrics.")
    parser.add_argument("dataset", type=Path, help="Path to a JSONL evaluation dataset.")
    parser.add_argument("--k", type=int, default=5, help="Cutoff for recall/nDCG metrics.")
    args = parser.parse_args()

    examples = load_jsonl_dataset(args.dataset)
    print(json.dumps(build_report(examples, k=args.k), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

