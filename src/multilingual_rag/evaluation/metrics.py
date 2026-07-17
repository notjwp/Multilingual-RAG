"""Deterministic retrieval and generation evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(expected_ids: Sequence[str], retrieved_ids: Sequence[str], *, k: int) -> float:
    """Return recall@k for expected and retrieved document IDs."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    expected = set(expected_ids)
    if not expected:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    return len(expected & retrieved) / len(expected)


def reciprocal_rank(expected_ids: Sequence[str], retrieved_ids: Sequence[str]) -> float:
    """Return reciprocal rank for the first relevant retrieved document."""
    expected = set(expected_ids)
    if not expected:
        return 0.0
    for index, document_id in enumerate(retrieved_ids, start=1):
        if document_id in expected:
            return 1.0 / index
    return 0.0


def dcg_at_k(expected_ids: Sequence[str], retrieved_ids: Sequence[str], *, k: int) -> float:
    """Return binary discounted cumulative gain at k."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    expected = set(expected_ids)
    return sum(
        1.0 / math.log2(index + 1)
        for index, document_id in enumerate(retrieved_ids[:k], start=1)
        if document_id in expected
    )


def ndcg_at_k(expected_ids: Sequence[str], retrieved_ids: Sequence[str], *, k: int) -> float:
    """Return normalized discounted cumulative gain at k."""
    ideal_hits = min(len(set(expected_ids)), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg_at_k(expected_ids, retrieved_ids, k=k) / ideal_dcg


def language_match_rate(
    expected_languages: Sequence[str | None],
    answer_languages: Sequence[str | None],
) -> float:
    """Return the share of examples where answer language matches the expected language."""
    pairs = [
        (expected, actual)
        for expected, actual in zip(expected_languages, answer_languages, strict=True)
        if expected is not None
    ]
    if not pairs:
        return 0.0
    return sum(1 for expected, actual in pairs if expected == actual) / len(pairs)


def citation_precision(cited_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Fraction of cited sources that are relevant. 1.0 when nothing is cited (vacuously exact)."""
    cited = set(cited_ids)
    if not cited:
        return 1.0
    return len(cited & set(relevant_ids)) / len(cited)


def citation_recall(cited_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Fraction of relevant sources that were cited. 0.0 when there is nothing relevant to cite."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(cited_ids) & relevant) / len(relevant)
