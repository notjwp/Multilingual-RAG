"""Romanized-Hindi retrieval eval — reconstructs the spike's A/B/C measurement.

Indexes the native-Devanagari XQuAD-hi corpus (gold + distractors) with bge-m3, then scores five
query conditions against it:

  * native            — the Devanagari question (sanity gate, ~0.90)
  * romanized-raw     — the question romanized (how people type); the floor (~0.20)
  * transliterated    — romanized then transliterated back to Devanagari (the fix)
  * shipped           — transliterate only when detected as romanized Hindi, else search raw
  * agentic           — the real agent graph: route -> retrieve -> grade -> (repair -> retrieve)

`shipped` is what the retrieval layer does on its own; `agentic` is what actually ships now, and
the gap between them is what the grade-and-repair cycle buys. `agentic` drives the real
``RagGraph`` — the same nodes, routers and repair logic the API uses.

**Generation is stubbed** in the agentic condition, deliberately: this measures retrieval, and
stubbing keeps the eval free and offline like every other condition here. The stub's ``acomplete``
returns an empty string, so an LLM-rewrite repair degrades to a no-op retry
(``clean_standalone_query`` falls back to the original query) rather than searching a fabricated
one. In practice the interesting strategy on this corpus is ``raw_fallback``, which costs nothing.

One transliteration call per query. Retrieval is local/free; only transliteration may use the
network. There is deliberately **no PASS/FAIL bar**: the historical 0.747 was derived before the
romanization fix (see ``evaluation/romanization.py``) and is not comparable to anything this
script now prints. Re-derive one on the full corpus before reinstating a gate.

Usage:
    python scripts/eval_romanized.py --sample 150            # google (default), full distractors
    python scripts/eval_romanized.py --provider rule-based   # compare a local backend
    python scripts/eval_romanized.py --relevance-threshold 0.5   # tune the repair trigger
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import time
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from statistics import mean

from multilingual_rag.agent.factory import build_rag_graph, build_stream_client
from multilingual_rag.agent.grading.factory import build_relevance_grader
from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import ConversationTurn, RetrievalContext
from multilingual_rag.embeddings.bge_embeddings import BgeM3EmbeddingProvider
from multilingual_rag.evaluation.datasets import load_xquad_corpus
from multilingual_rag.evaluation.harness import EVAL_USER_ID, ingest_documents
from multilingual_rag.evaluation.metrics import recall_at_k, reciprocal_rank
from multilingual_rag.evaluation.romanization import romanize
from multilingual_rag.generation.base import StreamClient
from multilingual_rag.retrieval.base import Retriever
from multilingual_rag.retrieval.routing import LanguageRoute
from multilingual_rag.retrieval.service import RetrievalService
from multilingual_rag.transliteration.detect import detect_target_language
from multilingual_rag.transliteration.factory import build_transliterator
from multilingual_rag.vectorstores.base import VectorFilter


class StubStreamClient:
    """Generation isn't measured here, so it's stubbed — the eval stays free and offline.

    ``acomplete`` returns "" so ``clean_standalone_query`` falls back to the original query: an
    LLM-rewrite repair becomes a no-op retry rather than a search for a fabricated query.
    """

    async def astream_completion(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        history: Sequence[ConversationTurn] = (),
    ) -> AsyncIterator[str]:
        yield "[generation stubbed for retrieval eval]"

    async def acomplete(self, *, model: str, system: str, prompt: str) -> str:
        return ""


class CountingRetriever:
    """Wraps the real retriever to count retrieval attempts, so repairs are observable."""

    def __init__(self, inner: Retriever) -> None:
        self.inner = inner
        self.attempts = 0

    def route(
        self,
        query: str,
        *,
        force_language: str | None = None,
        skip_transliteration: bool = False,
    ) -> LanguageRoute:
        return self.inner.route(
            query, force_language=force_language, skip_transliteration=skip_transliteration
        )

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int | None = None,
        filters: VectorFilter | None = None,
        route: LanguageRoute | None = None,
    ) -> RetrievalContext:
        self.attempts += 1
        return self.inner.retrieve(
            query,
            user_id=user_id,
            session_id=session_id,
            top_k=top_k,
            filters=filters,
            route=route,
        )


# romanize()/rule_romanize() live in the package (multilingual_rag.evaluation.romanization) rather
# than here, so the test suite and mypy --strict cover them: how the romanized queries are
# synthesized decides whether this evaluation is honest at all. See that module's docstring.


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Romanized-Indic retrieval evaluation.")
    parser.add_argument("--lang", default="hi", choices=["hi", "kn", "te"])
    parser.add_argument(
        "--corpus-dir", "--xquad-dir", dest="corpus_dir", type=Path, default=Path("data/eval/xquad")
    )
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
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=None,
        help="Cosine floor below which the agent repairs and retries (default: from Settings).",
    )
    parser.add_argument(
        "--max-repairs", type=int, default=None, help="Repair attempts per query (default: 1)."
    )
    args = parser.parse_args()

    corpus = load_xquad_corpus(args.corpus_dir, (args.lang,), sample=args.distractor_cap)
    queries = corpus.queries[: args.sample]
    settings = Settings(environment="test", transliteration_provider=args.provider)
    agent_overrides: dict[str, object] = {}
    if args.relevance_threshold is not None:
        agent_overrides["relevance_score_threshold"] = args.relevance_threshold
    if args.max_repairs is not None:
        agent_overrides["agent_max_repairs"] = args.max_repairs
    if agent_overrides:
        settings = settings.model_copy(update=agent_overrides)
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
        det = settings.transliteration_detector
        print(f"[{args.lang}] indexed {n_docs} native-script docs; scoring {len(queries)} queries "
              f"(k={args.k}, provider={args.provider}, detector={det})")
        print(f"       agent: grader={settings.relevance_grader} "
              f"threshold={settings.relevance_score_threshold} "
              f"max_repairs={settings.agent_max_repairs}\n")

        def search(text: str) -> tuple[str, ...]:
            embedding = embedder.embed_query(text)
            results = store.search(embedding, user_id=EVAL_USER_ID, top_k=args.k)
            return tuple(r.document_id for r in results)

        # The real graph over the same store. Generation is stubbed (see module docstring), but
        # the grader is the configured one: with RELEVANCE_GRADER=llm it makes real calls, which
        # is the only way to measure whether the repair loop is worth its cost.
        stub_client = StubStreamClient()
        grader_client: StreamClient = stub_client
        if settings.relevance_grader == "llm":
            grader_client = build_stream_client(settings)
        retriever = CountingRetriever(
            RetrievalService(
                settings,
                embedding_provider=embedder,
                vector_store=store,
                transliterator=transliterator,
            )
        )
        graph = build_rag_graph(
            settings,
            retriever=retriever,
            client=stub_client,
            grader=build_relevance_grader(settings, client=grader_client),
        )

        def run_agent(text: str) -> tuple[tuple[str, ...], int]:
            """Drive the real graph; return the finally-retrieved docs and the attempt count."""
            retriever.attempts = 0
            result = asyncio.run(graph.answer(text, user_id=EVAL_USER_ID, top_k=args.k))
            docs = tuple(r.document_id for r in result.context.results)
            return docs, retriever.attempts

        conditions = ("native", "romanized-raw", "transliterated", "shipped", "agentic")
        scores: dict[str, dict[str, list[float]]] = {
            cond: {"recall": [], "rr": []} for cond in conditions
        }
        n_detected = 0
        n_repaired = 0
        roman_stats: Counter[str] = Counter()
        for i, query in enumerate(queries):
            expected = query.expected_document_ids
            roman = romanize(query.question, args.lang, stats=roman_stats)
            translit = transliterator.transliterate(roman, target_language=args.lang)
            if args.pace:
                time.sleep(args.pace)

            retrieved = {
                "native": search(query.question),
                "romanized-raw": search(roman),
                "transliterated": search(translit),
            }
            # The shipped path: transliterate only when the detector identifies this language,
            # else search the raw query untouched (exactly what RetrievalService does).
            detected = (
                detect_target_language(
                    roman, (args.lang,), detector=settings.transliteration_detector
                )
                == args.lang
            )
            n_detected += detected
            retrieved["shipped"] = (
                retrieved["transliterated"] if detected else retrieved["romanized-raw"]
            )
            # The agent sees the same romanized query a user would type, and decides for itself.
            retrieved["agentic"], attempts = run_agent(roman)
            n_repaired += attempts > 1

            for cond, docs in retrieved.items():
                scores[cond]["recall"].append(recall_at_k(expected, docs, k=args.k))
                scores[cond]["rr"].append(reciprocal_rank(expected, docs))
            if (i + 1) % 25 == 0:
                print(f"  ...{i + 1}/{len(queries)}")

    words = roman_stats["hit"] + roman_stats["oov"]
    if words:
        share = roman_stats["hit"] / words
        print(f"\nquery words from HUMAN romanizations: {roman_stats['hit']}/{words} "
              f"({share:.1%})")
        if roman_stats["oov"]:
            # Disclosed, not hidden: the remainder is rule-based output, which the `rule-based`
            # adapter can invert exactly. That is a residual advantage for it, bounded by this
            # figure. Grow data/eval/romanization_hi.json to shrink it.
            print(f"  remaining {roman_stats['oov']} words fell back to sanscript — a residual "
                  f"edge for the rule-based adapter")
    else:
        print("\nWARNING: no human romanization lexicon found — queries are pure sanscript "
              "output, which unfairly flatters the rule-based adapter. "
              "Run scripts/build_romanization_lexicon.py.")
    print(f"romanized-{args.lang} detected: {n_detected}/{len(queries)} "
          f"({n_detected / len(queries):.1%})")
    print(f"agent repaired:        {n_repaired}/{len(queries)} "
          f"({n_repaired / len(queries):.1%}) — queries where retrieval was graded weak")
    native_recall = _mean(scores["native"]["recall"]) or 1e-9
    print(f"\n{'condition':<16}{'recall@' + str(args.k):>10}{'MRR':>8}{'retention':>11}")
    print("-" * 45)
    for cond in conditions:
        recall = _mean(scores[cond]["recall"])
        rr = _mean(scores[cond]["rr"])
        retention = recall / native_recall
        print(f"{cond:<16}{recall:>10.3f}{rr:>8.3f}{retention:>11.3f}")

    shipped_ret = _mean(scores["shipped"]["recall"]) / native_recall
    agentic_ret = _mean(scores["agentic"]["recall"]) / native_recall
    # The historical bar was 0.747, derived when queries were generated by sanscript — the same
    # library as the rule-based adapter under test (see evaluation/romanization.py). Numbers from
    # that harness are not comparable to these, so printing PASS/FAIL against it would be
    # meaningless. State the retention; let a re-derived bar replace this.
    print(f"\nshipped retention: {shipped_ret:.3f}   (historical bar 0.747 NOT comparable — it "
          f"predates the romanization fix and needs re-deriving)")
    delta = _mean(scores["agentic"]["recall"]) - _mean(scores["shipped"]["recall"])
    verdict = "helps" if delta > 0 else ("no change" if delta == 0 else "HURTS")
    print(f"Agent vs shipped:      {delta:+.3f} recall@{args.k} "
          f"({agentic_ret:.3f} vs {shipped_ret:.3f} retention) — {verdict}")
    if n_repaired == 0 and settings.relevance_score_threshold == 0.0:
        print("  note: repair never fired, as expected — the default floor flags only *empty*\n"
              "        retrievals. Positive floors were measured and lost (score_threshold.py);\n"
              "        for a repair loop that actually fires, use RELEVANCE_GRADER=llm.")
    elif n_repaired == 0:
        print(f"  note: repair never fired at threshold {settings.relevance_score_threshold} — "
              "nothing scored below it.")


if __name__ == "__main__":
    main()
