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

Output: ``data/eval/romanization_hi.json`` — native word -> most-attested human spelling, filtered
to the XQuAD-hi question vocabulary so the artifact stays small and committable. Words still
missing fall back to sanscript at eval time, and the eval prints that residual rate.

Usage:
    python scripts/build_romanization_lexicon.py
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

# Pinned so regeneration is reproducible, matching build_eval_corpus.py's convention.
DAKSHINA_REPO = "Anvesh-Lankala/Copy_Dakshina_Google_research_dataset"
DAKSHINA_FILE = "data/hi-00000-of-00001.parquet"
PARALLEL_REPO = "ar5entum/hindi-english-roman-devnagiri-transliteration-corpus"

OUT_PATH = Path(__file__).parents[1] / "data" / "eval" / "romanization_hi.json"
QUERIES_PATH = Path(__file__).parents[1] / "data" / "eval" / "xquad" / "queries_hi.jsonl"

DEVANAGARI = re.compile(r"[ऀ-ॿ]+")
# Human romanizations are Latin + digits; reject rows that leaked native script or punctuation.
ROMAN_OK = re.compile(r"^[a-z0-9.'-]+$")


def _clean(word: str) -> str | None:
    word = word.strip().lower().strip(".,!?;:\"'()")
    return word if word and ROMAN_OK.match(word) else None


def from_dakshina() -> dict[str, Counter[str]]:
    """Per-word attested spellings, straight from Dakshina's lexicon_mapping column."""
    ds = load_dataset(DAKSHINA_REPO, data_files=DAKSHINA_FILE, split="train")
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
            if not DEVANAGARI.fullmatch(native):
                continue  # skip punctuation, digits, and Latin already on the native side
            cleaned = _clean(roman)
            if cleaned:
                lexicon.setdefault(native, Counter())[cleaned] += 1
    return lexicon


def query_vocabulary() -> set[str]:
    """Every Devanagari token appearing in the XQuAD-hi questions."""
    vocab: set[str] = set()
    with QUERIES_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                vocab.update(DEVANAGARI.findall(json.loads(line)["question"]))
    return vocab


def main() -> None:
    print("loading Dakshina (human per-word attestations)...")
    dakshina = from_dakshina()
    print(f"  {len(dakshina)} words")

    print("loading the parallel corpus (word-aligned, equal-length rows only)...")
    parallel = from_parallel_corpus()
    print(f"  {len(parallel)} words")

    vocab = query_vocabulary()
    print(f"\nXQuAD-hi question vocabulary: {len(vocab)} distinct Devanagari tokens")

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

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(lexicon, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8"
    )
    covered = len(lexicon)
    print(f"\nwrote {OUT_PATH.relative_to(Path(__file__).parents[1])}")
    print(f"  {covered}/{len(vocab)} tokens covered ({covered / len(vocab):.1%})")
    print(f"    from Dakshina        : {from_d}")
    print(f"    from parallel corpus : {from_p}")
    print(f"  size: {OUT_PATH.stat().st_size / 1024:.0f} KB")
    print("\nUncovered words fall back to sanscript at eval time; the eval reports that rate.")


if __name__ == "__main__":
    main()
