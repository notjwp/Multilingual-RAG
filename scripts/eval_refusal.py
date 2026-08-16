"""Measure whether the agent answers questions its documents cannot answer.

**The gap this closes.** Every existing metric runs on XQuAD, where *every question has an answer
in the corpus*. There is no such thing as a correct refusal there, so no configuration can be
scored on refusal quality and the harness is structurally blind to the worst failure mode:
answering from parametric knowledge and attaching a citation to an unrelated passage.

That failure is real, not hypothetical. Manual testing against a chat holding one romanized-Hindi
document about health produced::

    Q: bharat ka rajdhaan kya hai?     (nothing in the corpus mentions Delhi)
    A: Bharat ka rajdhaan Dilli hai. [1]      <- cited the health document

Two query sets, so both halves of the tradeoff are visible at once:

===============  ==========================  ===================================================
set              questions vs corpus         correct behaviour
===============  ==========================  ===================================================
answerable       XQuAD-hi vs XQuAD-hi        answer, citing something
unanswerable     XQuAD-hi vs Kannada corpus  refuse, citing nothing
===============  ==========================  ===================================================

Hindi questions against a Kannada corpus are genuinely unanswerable while staying in-domain, and
retrieval still returns a full, plausible-looking, entirely wrong top-k — exactly the condition
that produced the hallucination. No new datasets; both corpora are already committed.

**Refusal is detected structurally, as zero resolved citations** — never by matching phrases,
which would break across the languages this project exists to serve. ``answer_citations`` returns
``()`` when no marker resolves, and ``generate_no_context`` guarantees it.

Generation is **real** here (unlike eval_romanized.py, which stubs it), so budget one provider
call per query and keep ``--sample`` modest.

Usage:
    python scripts/eval_refusal.py --sample 20
    python scripts/eval_refusal.py --sample 20 --grader llm         # measure the tradeoff
    python scripts/eval_refusal.py --sample 20 --grounding-gate     # judge the answer, not the
                                                                    # retrieval
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from multilingual_rag.agent.factory import build_rag_graph, build_stream_client
from multilingual_rag.agent.grading.factory import build_relevance_grader
from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.embeddings.bge_embeddings import BgeM3EmbeddingProvider
from multilingual_rag.evaluation.datasets import load_xquad_corpus
from multilingual_rag.evaluation.harness import EVAL_USER_ID, ingest_documents
from multilingual_rag.evaluation.romanization import romanize
from multilingual_rag.retrieval.service import RetrievalService
from multilingual_rag.transliteration.factory import build_transliterator
from multilingual_rag.vectorstores.chroma_store import ChromaVectorStore

XQUAD_DIR = Path("data/eval/xquad")
INDIC_DIR = Path("data/eval/indic")


@dataclass
class Outcome:
    answered: int = 0
    refused: int = 0
    errored: int = 0

    @property
    def total(self) -> int:
        """Only queries that actually produced a verdict; errors are excluded from the rates."""
        return self.answered + self.refused


def _run_set(
    *,
    label: str,
    corpus_dir: Path,
    corpus_lang: str,
    settings: Settings,
    embedder: BgeM3EmbeddingProvider,
    questions: list[str],
    distractor_cap: int,
    pace: float,
) -> Outcome:
    """Index one corpus, ask ``questions`` against it, and count answers vs refusals."""
    corpus = load_xquad_corpus(corpus_dir, (corpus_lang,), sample=distractor_cap)
    outcome = Outcome()

    with tempfile.TemporaryDirectory(prefix=f"refusal-{label}-", ignore_cleanup_errors=True) as tmp:
        scoped = settings.model_copy(
            update={"chroma_persist_directory": Path(tmp), "chroma_collection_name": "refusal"}
        )
        store = ChromaVectorStore(scoped)
        n_docs = ingest_documents(store, embedder, corpus.documents)
        client = build_stream_client(scoped)
        graph = build_rag_graph(
            scoped,
            retriever=RetrievalService(
                scoped,
                embedding_provider=embedder,
                vector_store=store,
                transliterator=build_transliterator(scoped),
            ),
            client=client,
            grader=build_relevance_grader(scoped, client=client),
        )
        print(f"\n[{label}] {n_docs} {corpus_lang} docs, {len(questions)} questions")

        for index, question in enumerate(questions):
            try:
                result = asyncio.run(graph.answer(question, user_id=EVAL_USER_ID, top_k=5))
            except AppError as exc:
                # One flaky call must not discard a multi-minute run. Both failure modes are real
                # on a free tier: `generation_rate_limited` (40 RPM, and the llm grader roughly
                # triples calls per query) and `generation_stream_corrupt` (a malformed SSE frame,
                # seen after ~30 healthy calls). Count it and carry on; errors are excluded from
                # the rates rather than silently scored as refusals, which would flatter the run.
                outcome.errored += 1
                print(f"  [{index + 1}] skipped: {exc.code}")
            else:
                # Structural, not phrase-based: no resolved citation == a refusal.
                if result.answer.citations:
                    outcome.answered += 1
                else:
                    outcome.refused += 1
            if pace:
                time.sleep(pace)
            if (index + 1) % 10 == 0:
                print(f"  ...{index + 1}/{len(questions)}")
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure answer-vs-refuse behaviour.")
    parser.add_argument("--sample", type=int, default=20, help="Questions per set.")
    parser.add_argument("--distractor-cap", type=int, default=1000)
    parser.add_argument(
        "--grader", choices=["score-threshold", "llm"], default=None,
        help="Override RELEVANCE_GRADER for this run.",
    )
    parser.add_argument(
        "--grounding-gate", action="store_true",
        help="Judge each drafted answer against its passages and refuse when unsupported. "
             "Adds one provider call per answered turn, on top of the grader's.",
    )
    parser.add_argument(
        "--pace", type=float, default=1.0,
        help="Seconds between queries. NIM free tier is 40 RPM and the llm grader "
             "roughly triples calls per query.",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings(environment="test")
    if args.grader:
        settings = settings.model_copy(update={"relevance_grader": args.grader})
    if args.grounding_gate:
        settings = settings.model_copy(update={"grounding_gate": True})
    print(f"grader={settings.relevance_grader} threshold={settings.relevance_score_threshold} "
          f"grounding_gate={settings.grounding_gate} model={settings.generation_model}")

    # The same romanized Hindi questions drive both sets; only the corpus changes.
    hi = load_xquad_corpus(XQUAD_DIR, ("hi",), sample=args.distractor_cap)
    questions = [romanize(q.question, "hi") for q in hi.queries[: args.sample]]

    embedder = BgeM3EmbeddingProvider()
    answerable = _run_set(
        label="answerable", corpus_dir=XQUAD_DIR, corpus_lang="hi", settings=settings,
        embedder=embedder, questions=questions, distractor_cap=args.distractor_cap,
        pace=args.pace,
    )
    unanswerable = _run_set(
        label="unanswerable", corpus_dir=INDIC_DIR, corpus_lang="kn", settings=settings,
        embedder=embedder, questions=questions, distractor_cap=args.distractor_cap,
        pace=args.pace,
    )

    hallucination_rate = unanswerable.answered / unanswerable.total if unanswerable.total else None
    false_refusal_rate = answerable.refused / answerable.total if answerable.total else None

    fr = f"{false_refusal_rate:.0%} false refusals" if false_refusal_rate is not None else ""
    hr = f"{hallucination_rate:.0%} hallucinated" if hallucination_rate is not None else ""
    print("\n" + "=" * 62)
    print(f"{'set':<16}{'answered':>10}{'refused':>10}   {'rate':<22}")
    print("-" * 62)
    print(f"{'answerable':<16}{answerable.answered:>10}{answerable.refused:>10}   {fr:<22}")
    if answerable.errored or unanswerable.errored:
        print(f"  (skipped: {answerable.errored} answerable, {unanswerable.errored} "
              f"unanswerable -- provider errors, excluded from the rates)")
    print(f"{'unanswerable':<16}{unanswerable.answered:>10}{unanswerable.refused:>10}   {hr:<22}")
    print("\nlower is better on both. They trade against each other — a configuration that never")
    print("answers scores 0% hallucinations and 100% false refusals, and is useless.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "grader": settings.relevance_grader,
                    "grounding_gate": settings.grounding_gate,
                    "relevance_score_threshold": settings.relevance_score_threshold,
                    "generation_model": settings.generation_model,
                    "questions_per_set": len(questions),
                    "answerable": {"answered": answerable.answered, "refused": answerable.refused},
                    "unanswerable": {
                        "answered": unanswerable.answered, "refused": unanswerable.refused
                    },
                    "hallucination_rate": hallucination_rate,
                    "false_refusal_rate": false_refusal_rate,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"\nreport written: {args.report}")


if __name__ == "__main__":
    main()
