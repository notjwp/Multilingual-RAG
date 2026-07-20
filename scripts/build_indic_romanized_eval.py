"""Build a romanized-retrieval eval for Kannada/Telugu (and hi) from Wikipedia.

There is no ready-made romanized kn/te Q&A corpus (IndicQA-romanized lacks them; FLORES-plus is
gated; IndicQA/FLORES-200 are script-based and unloadable in datasets 5.x). So we synthesize one
the same way the Hindi eval works: take native-script Wikipedia sentences as gold documents, use
their (synthetic, `indic_transliteration`) romanizations as queries, and add more native sentences
as distractors. Retrieval then tests whether romanized→transliterated recovers the native match.

Writes the same `gold_/queries_/distractors_{lang}.jsonl` layout as `build_eval_corpus.py`, so
`eval_romanized.py --lang kn --corpus-dir data/eval/indic` consumes it directly.

Usage:
    python scripts/build_indic_romanized_eval.py --langs kn te --gold 500 --distractors 2000
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "eval" / "indic"
WIKI_REVISION = "20231101"  # pinned Wikipedia dump config prefix
_SENT_END = re.compile(r"[.।॥?!]")
# Require the sentence to actually contain native script (not a stub of latin/numbers/links).
_NATIVE = {
    "hi": re.compile(r"[ऀ-ॿ]"),
    "kn": re.compile(r"[ಀ-೿]"),
    "te": re.compile(r"[ఀ-౿]"),
}


def first_sentence(text: str) -> str:
    collapsed = " ".join(text.split())
    match = _SENT_END.search(collapsed)
    return (collapsed[: match.start() + 1] if match else collapsed).strip()


def passages(lang: str, needed: int) -> list[str]:
    """Stream Wikipedia for `needed` native-script first-sentences (deduped, 30–300 chars)."""
    dataset = load_dataset(
        "wikimedia/wikipedia", f"{WIKI_REVISION}.{lang}", split="train", streaming=True
    )
    native = _NATIVE[lang]
    out: list[str] = []
    seen: set[str] = set()
    for row in dataset:
        sentence = first_sentence(str(row["text"]))
        if not (30 <= len(sentence) <= 300) or not native.search(sentence):
            continue
        if sentence in seen:
            continue
        seen.add(sentence)
        out.append(sentence)
        if len(out) >= needed:
            break
    return out


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the kn/te romanized retrieval eval.")
    parser.add_argument("--langs", nargs="+", default=["kn", "te"])
    parser.add_argument("--gold", type=int, default=500, help="Gold docs = queries per language.")
    parser.add_argument("--distractors", type=int, default=2000, help="Extra distractor docs/lang.")
    args = parser.parse_args()

    for lang in args.langs:
        print(f"[{lang}] collecting {args.gold + args.distractors} Wikipedia sentences...")
        pool = passages(lang, args.gold + args.distractors)
        gold, distractors = pool[: args.gold], pool[args.gold :]

        write_jsonl(OUT_DIR / f"gold_{lang}.jsonl",
                    [{"para_idx": i, "text": text} for i, text in enumerate(gold)])
        write_jsonl(OUT_DIR / f"queries_{lang}.jsonl",
                    [{"qid": f"{lang}-{i}", "question": text, "para_idx": i}
                     for i, text in enumerate(gold)])
        write_jsonl(OUT_DIR / f"distractors_{lang}.jsonl",
                    [{"docid": f"{lang}-dist-{i}", "text": text}
                     for i, text in enumerate(distractors)])
        print(f"[{lang}] wrote {len(gold)} gold/queries + {len(distractors)} distractors")

    print(f"\nCorpus written to {OUT_DIR}")


if __name__ == "__main__":
    main()
