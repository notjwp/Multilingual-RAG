"""Offline evaluation report CLI.

Two modes:
  * fixture (default): score precomputed ``retrieved_document_ids`` from a JSONL dataset.
  * ``--live``: run the real retrieval pipeline (bge-m3 + Chroma) over the XQuAD corpus and
    score the results. Free — bge-m3 is local. Needs the ``eval`` extra installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from multilingual_rag.evaluation.datasets import (
    EvaluationExample,
    load_jsonl_dataset,
    load_xquad_corpus,
)
from multilingual_rag.evaluation.metrics import (
    language_match_rate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

DEFAULT_XQUAD_DIR = Path("data/eval/xquad")


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


def run_fixture(dataset: Path, *, k: int) -> dict[str, Any]:
    return build_report(load_jsonl_dataset(dataset), k=k)


def run_live(
    directory: Path,
    languages: tuple[str, ...],
    *,
    k: int,
    sample: int | None,
) -> dict[str, Any]:
    """Run retrieval end-to-end with bge-m3 + Chroma over the XQuAD corpus."""
    # Imported lazily: bge-m3/Chroma pull heavy deps only needed for the live path.
    import tempfile

    from multilingual_rag.core.config import Settings
    from multilingual_rag.embeddings.bge_embeddings import BgeM3EmbeddingProvider
    from multilingual_rag.evaluation.harness import run_live_evaluation
    from multilingual_rag.vectorstores.chroma_store import ChromaVectorStore

    corpus = load_xquad_corpus(directory, languages, sample=sample)
    # ignore_cleanup_errors: Chroma keeps its HNSW files open, and Windows won't unlink a
    # locked file — the temp dir is best-effort cleaned rather than crashing the run.
    with tempfile.TemporaryDirectory(
        prefix="rag-eval-chroma-", ignore_cleanup_errors=True
    ) as tmp:
        settings = Settings(
            environment="test",
            chroma_persist_directory=Path(tmp),
            chroma_collection_name="eval",
        )
        examples = run_live_evaluation(
            settings=settings,
            embedding_provider=BgeM3EmbeddingProvider(),
            vector_store=ChromaVectorStore(settings),
            corpus=corpus,
            top_k=k,
        )
    report = build_report(examples, k=k)
    report["document_count"] = len(corpus.documents)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline RAG evaluation metrics.")
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        help="Fixture JSONL dataset (fixture mode). Omit when using --live.",
    )
    parser.add_argument("--k", type=int, default=5, help="Cutoff for recall/nDCG metrics.")
    parser.add_argument("--live", action="store_true", help="Run the real retrieval pipeline.")
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["en", "es", "zh", "th"],
        help="Languages for --live (default: all four).",
    )
    parser.add_argument(
        "--xquad-dir",
        type=Path,
        default=DEFAULT_XQUAD_DIR,
        help="XQuAD corpus directory for --live.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Cap distractors/queries per language for a fast --live smoke run.",
    )
    args = parser.parse_args()

    if args.live:
        report = run_live(
            args.xquad_dir, tuple(args.langs), k=args.k, sample=args.sample
        )
    else:
        if args.dataset is None:
            parser.error("a dataset path is required in fixture mode (or pass --live)")
        report = run_fixture(args.dataset, k=args.k)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
