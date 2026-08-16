"""Build the human-romanization lexicon used to synthesize realistic romanized eval queries.

**Why this exists.** `scripts/eval_romanized.py` needs romanized versions of the native-script
XQuAD questions, because no romanized XQuAD exists. It used to synthesize them with
`indic_transliteration.sanscript` (Devanagari -> IAST -> strip diacritics). That quietly rigged
the evaluation: the `rule-based` adapter under test is *also* `indic_transliteration.sanscript`,
so it was being scored on inverting the exact character-level transform that generated its input.
Measured, that inflated it to recall@5 0.950 against google's 0.700 on identical queries — a gap
that says nothing about either adapter's real quality.

It also produced spellings no human types. Real romanized Hindi keeps English loanwords in
English:

    native   जोश नॉर्मन ने कितने बॉल को इंटरसेप्ट किया?
    sanscript  josa narmana ne kitane bala ko imtarasepta kiya
    human      josh norman ne kitane ball ko intercept kiya

So the queries are now built from **human-written** romanizations, from two sources:

- **Dakshina** (Google Research; the mirror below carries its per-word `lexicon_mapping`, which is
  a crowd-collected set of attested spellings per native word).
- A **Hindi<->roman parallel corpus**, word-aligned on the rows where both sides tokenize to the
  same length. Noisier, so it only fills words Dakshina lacks.

Neither source involves `indic_transliteration`, so no adapter under test can be scored on
inverting its own scheme.

Output: ``data/eval/romanization_{lang}.json`` — native word -> most-attested human spelling,
filtered to that language's eval query vocabulary so each artifact stays small and committable.
Words still missing fall back to sanscript at eval time, and the eval prints that residual rate.

**Coverage is the acceptance criterion.** Below ~50% the majority of each query is still
sanscript output, the rule-based adapter keeps a material advantage, and that language's
numbers must stay marked directional rather than presented as corrected. kn/te draw on
Dakshina alone (no parallel corpus exists for them) against Wikipedia-sentence vocabulary,
so they are expected to score lower than hi.

Usage:
    python scripts/build_romanization_lexicon.py --langs hi kn te
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from multilingual_rag.evaluation.romanization import NATIVE_WORD

# Pinned so regeneration is reproducible, matching build_eval_corpus.py's convention.
# The mirror carries all twelve Dakshina languages; we build the three this project evaluates.
DAKSHINA_REPO = "Anvesh-Lankala/Copy_Dakshina_Google_research_dataset"

# Hindi-only. There is no equivalent kn/te parallel corpus, and substituting a different
# language's would be worse than lower coverage — kn/te use Dakshina alone.
PARALLEL_REPO = "ar5entum/hindi-english-roman-devnagiri-transliteration-corpus"

ROOT = Path(__file__).parents[1]

# Where each language's eval queries live. hi is XQuAD questions; kn/te are Wikipedia sentences
# synthesized by build_indic_romanized_eval.py, so their vocabulary is much broader.
QUERIES_PATH: dict[str, Path] = {
    "hi": ROOT / "data" / "eval" / "xquad" / "queries_hi.jsonl",
    "kn": ROOT / "data" / "eval" / "indic" / "queries_kn.jsonl",
    "te": ROOT / "data" / "eval" / "indic" / "queries_te.jsonl",
}

# Human romanizations are Latin + digits; reject rows that leaked native script or punctuation.
ROMAN_OK = re.compile(r"^[a-z0-9.'-]+$")


def _clean(word: str) -> str | None:
    word = word.strip().lower().strip(".,!?;:\"'()")
    return word if word and ROMAN_OK.match(word) else None


def from_dakshina(lang: str) -> dict[str, Counter[str]]:
    """Per-word attested spellings, straight from Dakshina's lexicon_mapping column."""
    ds = load_dataset(
        DAKSHINA_REPO, data_files=f"data/{lang}-00000-of-00001.parquet", split="train"
    )
    lexicon: dict[str, Counter[str]] = {}
    for row in ds:
        raw = row.get("lexicon_mapping")
        if not raw:
            continue
        try:
            mapping = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except (ValueError, SyntaxError):
            continue
        if not isinstance(mapping, dict):
            continue
        for native, romans in mapping.items():
            if not isinstance(romans, list):
                continue
            for roman in romans:
                cleaned = _clean(str(roman))
                if cleaned:
                    lexicon.setdefault(native, Counter())[cleaned] += 1
    return lexicon


def from_parallel_corpus() -> dict[str, Counter[str]]:
    """Word-align a roman<->Devanagari parallel corpus, keeping only equal-length rows.

    Positional alignment is only sound when both sides tokenize to the same count; anything else
    is discarded rather than guessed at.
    """
    ds = load_dataset(PARALLEL_REPO, split="train")
    lexicon: dict[str, Counter[str]] = {}
    for row in ds:
        native_tokens = str(row["devnagiri"]).split()
        roman_tokens = str(row["roman"]).split()
        if not native_tokens or len(native_tokens) != len(roman_tokens):
            continue
        for native, roman in zip(native_tokens, roman_tokens, strict=True):
            native = native.strip(".,!?;:\"'()–—")
            if not NATIVE_WORD.fullmatch(native):
                continue  # skip punctuation, digits, and Latin already on the native side
            cleaned = _clean(roman)
            if cleaned:
                lexicon.setdefault(native, Counter())[cleaned] += 1
    return lexicon


def query_vocabulary(lang: str) -> set[str]:
    """Every native-script token appearing in this language's eval queries."""
    path = QUERIES_PATH[lang]
    if not path.exists():
        raise SystemExit(f"no eval queries for {lang} at {path}")
    vocab: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                vocab.update(NATIVE_WORD.findall(json.loads(line)["question"]))
    return vocab


def build(lang: str) -> float:
    """Build one language's lexicon; return the fraction of its query vocabulary covered."""
    print(f"\n=== {lang} ===")
    print("loading Dakshina (human per-word attestations)...")
    dakshina = from_dakshina(lang)
    print(f"  {len(dakshina)} words")

    parallel: dict[str, Counter[str]] = {}
    if lang == "hi":
        print("loading the parallel corpus (word-aligned, equal-length rows only)...")
        parallel = from_parallel_corpus()
        print(f"  {len(parallel)} words")
    else:
        print("no parallel corpus for this language — Dakshina only")

    vocab = query_vocabulary(lang)
    print(f"eval query vocabulary: {len(vocab)} distinct native tokens")

    lexicon: dict[str, str] = {}
    from_d = from_p = 0
    for word in sorted(vocab):
        # Dakshina wins ties: it is curated per-word, the parallel corpus is positional guesswork.
        if word in dakshina:
            lexicon[word] = dakshina[word].most_common(1)[0][0]
            from_d += 1
        elif word in parallel:
            lexicon[word] = parallel[word].most_common(1)[0][0]
            from_p += 1

    out_path = ROOT / "data" / "eval" / f"romanization_{lang}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(lexicon, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8"
    )
    coverage = len(lexicon) / len(vocab) if vocab else 0.0
    print(f"wrote {out_path.relative_to(ROOT)}")
    print(f"  {len(lexicon)}/{len(vocab)} tokens covered ({coverage:.1%})")
    print(f"    from Dakshina        : {from_d}")
    print(f"    from parallel corpus : {from_p}")
    print(f"  size: {out_path.stat().st_size / 1024:.0f} KB")

    # Coverage is the whole point: below ~50% most of each query is still sanscript output and the
    # rule-based adapter keeps a material advantage, so the eval is only partly de-rigged.
    if coverage < 0.5:
        print(f"  WARNING: {coverage:.1%} coverage — the majority of each query still falls back "
              f"to sanscript.\n           {lang} numbers stay DIRECTIONAL, not corrected.")
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Build human-romanization lexicons.")
    parser.add_argument(
        "--langs", nargs="+", default=["hi"], choices=sorted(QUERIES_PATH),
        help="Languages to build (default: hi).",
    )
    args = parser.parse_args()

    results = {lang: build(lang) for lang in args.langs}

    print("\n=== coverage summary ===")
    for lang, coverage in results.items():
        verdict = "usable" if coverage >= 0.5 else "TOO LOW — keep numbers directional"
        print(f"  {lang}: {coverage:.1%}  {verdict}")
    print("\nUncovered words fall back to sanscript at eval time; the eval reports that rate.")


if __name__ == "__main__":
    main()
