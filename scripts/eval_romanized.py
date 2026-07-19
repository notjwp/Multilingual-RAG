"""Romanized-Hindi retrieval eval — reconstructs the spike's A/B/C measurement.

Indexes the native-Devanagari XQuAD-hi corpus (gold + distractors) with bge-m3, then scores four
query conditions against it:

  * native            — the Devanagari question (sanity gate, ~0.90)
  * romanized-raw     — the question romanized (how people type); the floor (~0.20)
  * transliterated    — romanized then transliterated back to Devanagari (the fix)
  * shipped           — the production path: transliterate only when detected as romanized Hindi
                        (is_romanized_indic), else search the raw query untouched

One transliteration call per query. Retrieval is local/free; only transliteration may use the
network. Acceptance: shipped retention (recall@k / native) >= ~0.747.

Usage:
    python scripts/eval_romanized.py --sample 150            # google (default), full distractors
    python scripts/eval_romanized.py --provider rule-based   # compare a local backend
"""

from __future__ import annotations

import argparse
import tempfile
import time
import unicodedata
from pathlib import Path
from statistics import mean

from indic_transliteration import sanscript  # type: ignore[import-untyped]
from indic_transliteration.sanscript import (
    transliterate as _sanscript,  # type: ignore[import-untyped]
)

from multilingual_rag.core.config import Settings
from multilingual_rag.embeddings.bge_embeddings import BgeM3EmbeddingProvider
from multilingual_rag.evaluation.datasets import load_xquad_corpus
from multilingual_rag.evaluation.harness import EVAL_USER_ID, ingest_documents
from multilingual_rag.evaluation.metrics import recall_at_k, reciprocal_rank
from multilingual_rag.transliteration.detect import is_romanized_indic
from multilingual_rag.transliteration.factory import build_transliterator


def romanize(devanagari: str) -> str:
    """Devanagari -> diacritic-stripped ASCII romanization (how people actually type).

    IAST keeps every phoneme but marks long vowels/retroflexes with diacritics; stripping them
    (bhārata -> bharata) is deliberately lossy, mirroring real keyboard input.
    """
    iast = _sanscript(devanagari, sanscript.DEVANAGARI, sanscript.IAST)
    decomposed = unicodedata.normalize("NFKD", iast)
    # Keep ASCII only: drop combining diacritics AND any Devanagari matra sanscript left
    # untranslated (e.g. the candra-O in foreign loanwords), matching real pure-ASCII typing.
    ascii_only = "".join(
        ch for ch in decomposed if ord(ch) < 128 and not unicodedata.combining(ch)
    )
    return ascii_only.lower()


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Romanized-Hindi retrieval evaluation.")
    parser.add_argument("--xquad-dir", type=Path, default=Path("data/eval/xquad"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sample", type=int, default=150, help="Queries to score (network-bound).")
    parser.add_argument("--distractor-cap", type=int, default=None, help="Cap distractors (speed).")
    parser.add_argument(
        "--provider",
        default="google",
        choices=["google", "indicxlit", "rule-based", "llm"],
        help="Transliteration backend to evaluate (default: the shipped google).",
    )
    parser.add_argument("--pace", type=float, default=0.0, help="Seconds between transliterations.")
    args = parser.parse_args()

    corpus = load_xquad_corpus(args.xquad_dir, ("hi",), sample=args.distractor_cap)
    queries = corpus.queries[: args.sample]
    settings = Settings(environment="test", transliteration_provider=args.provider)
    transliterator = build_transliterator(settings)
    assert transliterator is not None

    embedder = BgeM3EmbeddingProvider()
    with tempfile.TemporaryDirectory(prefix="rag-romanized-", ignore_cleanup_errors=True) as tmp:
        from multilingual_rag.vectorstores.chroma_store import ChromaVectorStore

        settings = settings.model_copy(
            update={"chroma_persist_directory": Path(tmp), "chroma_collection_name": "romanized"}
        )
        store = ChromaVectorStore(settings)
        n_docs = ingest_documents(store, embedder, corpus.documents)
        print(f"indexed {n_docs} native-Devanagari docs; scoring {len(queries)} queries "
              f"(k={args.k}, provider={args.provider})\n")

        def search(text: str) -> tuple[str, ...]:
            embedding = embedder.embed_query(text)
            results = store.search(embedding, user_id=EVAL_USER_ID, top_k=args.k)
            return tuple(r.document_id for r in results)

        conditions = ("native", "romanized-raw", "transliterated", "shipped")
        scores: dict[str, dict[str, list[float]]] = {
            cond: {"recall": [], "rr": []} for cond in conditions
        }
        n_detected = 0
        for i, query in enumerate(queries):
            expected = query.expected_document_ids
            roman = romanize(query.question)
            translit = transliterator.transliterate(roman, target_language="hi")
            if args.pace:
                time.sleep(args.pace)

            retrieved = {
                "native": search(query.question),
                "romanized-raw": search(roman),
                "transliterated": search(translit),
            }
            # The shipped path: transliterate only when detected as romanized Hindi, else search
            # the raw query untouched (exactly what RetrievalService does in production).
            detected = is_romanized_indic(roman, ("hi",))
            n_detected += detected
            retrieved["shipped"] = (
                retrieved["transliterated"] if detected else retrieved["romanized-raw"]
            )

            for cond, docs in retrieved.items():
                scores[cond]["recall"].append(recall_at_k(expected, docs, k=args.k))
                scores[cond]["rr"].append(reciprocal_rank(expected, docs))
            if (i + 1) % 25 == 0:
                print(f"  ...{i + 1}/{len(queries)}")

    print(f"\nromanized-Hindi detected: {n_detected}/{len(queries)} "
          f"({n_detected / len(queries):.1%})")
    native_recall = _mean(scores["native"]["recall"]) or 1e-9
    print(f"\n{'condition':<16}{'recall@' + str(args.k):>10}{'MRR':>8}{'retention':>11}")
    print("-" * 45)
    for cond in conditions:
        recall = _mean(scores[cond]["recall"])
        rr = _mean(scores[cond]["rr"])
        retention = recall / native_recall
        print(f"{cond:<16}{recall:>10.3f}{rr:>8.3f}{retention:>11.3f}")

    shipped_ret = _mean(scores["shipped"]["recall"]) / native_recall
    bar = 0.747
    print(f"\nAcceptance (shipped retention >= {bar}): "
          f"{'PASS' if shipped_ret >= bar else 'FAIL'} ({shipped_ret:.3f})")


if __name__ == "__main__":
    main()
