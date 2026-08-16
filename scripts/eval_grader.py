"""Measure whether an LLM is good enough to be the agent's relevance judge.

**Why this exists.** The agent's repair loop only fires when the grader calls a retrieval weak, so
the grader's error rate *is* the loop's usefulness. The default judge model
(`meta/llama-3.1-8b-instruct`) was measured at **81% false alarms on correct retrievals** — it
calls almost everything weak, which is why `RELEVANCE_GRADER` defaults to the free
`score-threshold` instead.

Do not swap in a different model and hope. That mistake — building on an unvalidated measurement —
is what produced and then destroyed the `retransliterate` strategy earlier in this project. Run
this first, record the numbers for every model tried, and only adopt one that clears the bar.

**The two numbers that matter**, from the grader's point of view on real retrievals:

- *false alarms* — retrievals that actually found the gold document but were graded weak. These
  cost the user a pointless retry and, before the never-regress rule, cost them the right answer.
- *catches* — retrievals that genuinely missed and were graded weak. These are the only ones a
  repair can help.

**Both numbers are required, and that is not pedantry.** A judge that flags everything scores 100%
catches and is useless. A judge that flags *nothing* scores 0% false alarms and is equally useless
— and that is the likelier trap here, because ``LlmRelevanceGrader`` deliberately fails open, so a
model that times out or errors on every call grades everything relevant and would sail past a
false-alarm bar alone. NVIDIA NIM's larger free-tier models are known to hang (see
``docs/progress.md`` on the 70B), which makes this a live failure mode rather than a hypothetical.

**Bar: false alarms < 20% AND catches >= 50%.**

Usage:
    python scripts/eval_grader.py --model meta/llama-3.1-8b-instruct
    python scripts/eval_grader.py --model meta/llama-3.3-70b-instruct --sample 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from multilingual_rag.agent.factory import build_stream_client
from multilingual_rag.agent.grading.llm import LlmRelevanceGrader
from multilingual_rag.core.config import Settings
from multilingual_rag.embeddings.bge_embeddings import BgeM3EmbeddingProvider
from multilingual_rag.evaluation.datasets import load_xquad_corpus
from multilingual_rag.evaluation.harness import EVAL_USER_ID, ingest_documents
from multilingual_rag.evaluation.romanization import romanize
from multilingual_rag.transliteration.factory import build_transliterator
from multilingual_rag.vectorstores.chroma_store import ChromaVectorStore

FALSE_ALARM_BAR = 0.20
CATCH_BAR = 0.50


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure an LLM's fitness as a relevance judge.")
    parser.add_argument("--model", required=True, help="Provider model id to test as the judge.")
    parser.add_argument("--lang", default="hi", choices=["hi", "kn", "te"])
    parser.add_argument(
        "--corpus-dir", type=Path, default=Path("data/eval/xquad")
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sample", type=int, default=20, help="Queries to judge (1 call each).")
    parser.add_argument("--distractor-cap", type=int, default=3000)
    parser.add_argument("--report", type=Path, default=None, help="Write results as JSON here.")
    args = parser.parse_args()

    settings = Settings(environment="test")
    corpus = load_xquad_corpus(args.corpus_dir, (args.lang,), sample=args.distractor_cap)
    queries = corpus.queries[: args.sample]
    transliterator = build_transliterator(settings)
    assert transliterator is not None

    embedder = BgeM3EmbeddingProvider()
    grader = LlmRelevanceGrader(client=build_stream_client(settings), model=args.model)

    with tempfile.TemporaryDirectory(prefix="grader-eval-", ignore_cleanup_errors=True) as tmp:
        scoped = settings.model_copy(
            update={"chroma_persist_directory": Path(tmp), "chroma_collection_name": "grader"}
        )
        store = ChromaVectorStore(scoped)
        n_docs = ingest_documents(store, embedder, corpus.documents)
        print(f"judge={args.model}  docs={n_docs}  queries={len(queries)}\n")

        false_alarms = catches = hits = misses = graded_weak = 0
        for index, query in enumerate(queries):
            roman = romanize(query.question, args.lang)
            native = transliterator.transliterate(roman, target_language=args.lang)
            results = store.search(
                embedder.embed_query(native), user_id=EVAL_USER_ID, top_k=args.k
            )
            # Ground truth: did this retrieval actually surface the gold document?
            hit = any(r.document_id in query.expected_document_ids for r in results)
            grade = asyncio.run(grader.grade(query=roman, results=results))

            hits += hit
            misses += not hit
            graded_weak += not grade.relevant
            if not grade.relevant and hit:
                false_alarms += 1
            if not grade.relevant and not hit:
                catches += 1
            if (index + 1) % 10 == 0:
                print(f"  ...{index + 1}/{len(queries)}")

    false_alarm_rate = false_alarms / hits if hits else None
    catch_rate = catches / misses if misses else None
    print(f"\nretrievals that actually hit : {hits}/{len(queries)}")
    print(f"graded weak by the judge     : {graded_weak}/{len(queries)}")
    print(f"  false alarms (hit, graded weak) : {false_alarms}/{hits}"
          f"{f' = {false_alarm_rate:.0%}' if false_alarm_rate is not None else ''}")
    print(f"  catches (miss, graded weak)     : {catches}/{misses}"
          f"{f' = {catch_rate:.0%}' if catch_rate is not None else ''}")

    too_noisy = false_alarm_rate is None or false_alarm_rate >= FALSE_ALARM_BAR
    too_blind = catch_rate is None or catch_rate < CATCH_BAR
    verdict = "UNUSABLE" if (too_noisy or too_blind) else "usable"
    print(f"\nbar: false alarms < {FALSE_ALARM_BAR:.0%} AND catches >= {CATCH_BAR:.0%}"
          f"  ->  {verdict}")
    if too_noisy:
        print("  too noisy: it calls correct retrievals weak, so the loop fires on nothing.")
    if too_blind:
        # The fail-open signature: the grader returns relevant on any OpenAIError, so a model that
        # times out on every call looks flawless on false alarms while catching nothing.
        print("  too blind: it misses real failures. If false alarms are also ~0, suspect the "
              "model is erroring and the grader is failing open — check the provider.")
    if verdict == "UNUSABLE":
        print("  Do not set RELEVANCE_GRADER=llm with this model. Record the number and move on.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "lang": args.lang,
                    "queries": len(queries),
                    "documents": n_docs,
                    "hits": hits,
                    "graded_weak": graded_weak,
                    "false_alarms": false_alarms,
                    "false_alarm_rate": false_alarm_rate,
                    "catches": catches,
                    "catch_rate": catch_rate,
                    "false_alarm_bar": FALSE_ALARM_BAR,
                    "catch_bar": CATCH_BAR,
                    "verdict": verdict,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"report written: {args.report}")


if __name__ == "__main__":
    main()
