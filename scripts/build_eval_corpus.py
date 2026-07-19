"""Regenerate the cross-lingual evaluation corpus in `data/eval/xquad/`.

Gold set:    XQuAD, professionally translated and parallel across languages, so
             cross-lingual ground truth is exact (paragraph i in zh IS paragraph i in en).
Distractors: miracl-corpus Wikipedia passages, to make retrieval discriminative.
             Without them recall@5 is ~1.0 for every model and the measurement is useless.

`gold_*.jsonl` and `queries_*.jsonl` are committed (~1.8 MB) because they are the verified,
hard-to-rebuild artifact. `distractors_*.jsonl` (~51 MB) are NOT committed: they are pure
derivation from a pinned shard plus a fixed seed, so this script reproduces them exactly.

Both dataset revisions are pinned. SEED alone is not enough for reproducibility -- see the
comment on the revision constants below.

Originally written for the M0 thesis spike; see docs/m0/report.md for what it measured.

Usage:
    python scripts/build_eval_corpus.py
"""

from __future__ import annotations

import gzip
import json
import random
import re
from collections import OrderedDict
from itertools import islice
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Spanish is the Latin-script control. (German is also present in the corpus repo despite
# being absent from the dataset card's config list, but es serves the identical role.)
# Hindi (Devanagari) is the target for the romanized-query eval (docs/indic-romanized-spike.md).
LANGS = ("en", "es", "zh", "th", "hi")
N_DISTRACTORS = 20_000
POOL_SIZE = 100_000  # read this many, then randomly sample N_DISTRACTORS from it
SEED = 42

# Pinned so regeneration is byte-identical. SEED alone is NOT sufficient: rng.sample() is
# only deterministic if the pool it samples from is identical, which requires a fixed shard.
# Without these, an upstream republish silently yields a different corpus and the committed
# gold/queries stop matching the regenerated distractors.
XQUAD_REVISION = "51adfef1c1287aab1d2d91b5bead9bcfb9c68583"
MIRACL_REVISION = "d921ec7e349ce0d28daf30b2da9da5ee698bef0d"

DATA_DIR = Path(__file__).parents[1] / "data" / "eval" / "xquad"


def clean(text: str) -> str:
    """Strip BOM/zero-width noise. XQuAD's es and th contexts start with a literal U+FEFF,
    which would otherwise be embedded as a junk leading token."""
    return text.replace("﻿", "").replace("​", "").strip()


def normalize(text: str) -> str:
    """Normalize for duplicate detection only — not for embedding."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load_xquad(lang: str) -> tuple[list[str], list[dict[str, object]]]:
    """Return (ordered unique paragraphs, queries) for one XQuAD language.

    XQuAD has 240 paragraphs and 1190 questions, so contexts repeat. Paragraph identity
    is positional: the Nth unique context in zh is the translation of the Nth in en.
    """
    rows = load_dataset(
        "google/xquad",
        f"xquad.{lang}",
        split="validation",
        revision=XQUAD_REVISION,
    )

    paragraphs: OrderedDict[str, int] = OrderedDict()
    queries: list[dict[str, object]] = []
    for row in rows:
        context = clean(str(row["context"]))
        if context not in paragraphs:
            paragraphs[context] = len(paragraphs)
        queries.append(
            {
                "qid": str(row["id"]),
                "question": clean(str(row["question"])),
                "para_idx": paragraphs[context],
            }
        )
    return list(paragraphs.keys()), queries


def verify_alignment(per_lang: dict[str, tuple[list[str], list[dict[str, object]]]]) -> None:
    """Fail loudly if the parallel structure that makes ground truth exact is not intact.

    Everything downstream assumes paragraph i means the same thing in every language.
    If that is false, every cross-lingual number is meaningless.
    """
    reference = LANGS[0]
    ref_paragraphs, ref_queries = per_lang[reference]

    for lang in LANGS:
        paragraphs, queries = per_lang[lang]
        if len(paragraphs) != len(ref_paragraphs):
            raise SystemExit(
                f"Paragraph count mismatch: {lang} has {len(paragraphs)}, "
                f"{reference} has {len(ref_paragraphs)}"
            )
        if len(queries) != len(ref_queries):
            raise SystemExit(
                f"Query count mismatch: {lang} has {len(queries)}, "
                f"{reference} has {len(ref_queries)}"
            )

        ref_qids = [q["qid"] for q in ref_queries]
        qids = [q["qid"] for q in queries]
        if qids != ref_qids:
            raise SystemExit(f"Question id order differs between {lang} and {reference}")

        ref_mapping = [q["para_idx"] for q in ref_queries]
        mapping = [q["para_idx"] for q in queries]
        if mapping != ref_mapping:
            raise SystemExit(
                f"Question->paragraph mapping differs between {lang} and {reference}; "
                "positional alignment is broken"
            )

    print(
        f"  alignment OK: {len(ref_paragraphs)} paragraphs, {len(ref_queries)} queries, "
        f"identical qid order and paragraph mapping across {', '.join(LANGS)}"
    )


def load_distractors(lang: str, gold: list[str]) -> list[dict[str, str]]:
    """Sample distractor passages, dropping any that duplicate a gold paragraph.

    Reads the raw .jsonl.gz shard directly rather than via load_dataset: `datasets` 5.x
    removed script-based loaders and miracl-corpus ships a loading script.

    The dedup matters most for `en`: XQuAD contexts come from English Wikipedia via SQuAD,
    and miracl-corpus `en` is also Wikipedia, so a gold paragraph can genuinely reappear as
    a distractor. That would make the gold document ambiguous and corrupt en-> cells.
    """
    gold_normalized = {normalize(text) for text in gold}

    shard = hf_hub_download(
        "miracl/miracl-corpus",
        f"miracl-corpus-v1.0-{lang}/docs-0.jsonl.gz",
        repo_type="dataset",
        revision=MIRACL_REVISION,
    )

    pool: list[dict[str, str]] = []
    dropped = 0
    with gzip.open(shard, "rt", encoding="utf-8") as handle:
        for line in islice(handle, POOL_SIZE):
            row = json.loads(line)
            text = str(row["text"]).strip()
            if not text:
                continue
            if normalize(text) in gold_normalized:
                dropped += 1
                continue
            pool.append({"docid": str(row["docid"]), "text": text})

    rng = random.Random(SEED)
    distractors = rng.sample(pool, min(N_DISTRACTORS, len(pool)))
    print(
        f"  {len(distractors)} distractors sampled from {len(pool)} "
        f"({dropped} dropped as gold duplicates)"
    )
    return distractors


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    print("Loading XQuAD...")
    per_lang = {lang: load_xquad(lang) for lang in LANGS}
    verify_alignment(per_lang)

    for lang in LANGS:
        paragraphs, queries = per_lang[lang]
        write_jsonl(
            DATA_DIR / f"gold_{lang}.jsonl",
            [{"para_idx": i, "text": text} for i, text in enumerate(paragraphs)],
        )
        write_jsonl(DATA_DIR / f"queries_{lang}.jsonl", queries)

    for lang in LANGS:
        print(f"Sampling distractors for {lang}...")
        paragraphs, _ = per_lang[lang]
        write_jsonl(DATA_DIR / f"distractors_{lang}.jsonl", load_distractors(lang, paragraphs))

    print(f"\nCorpus written to {DATA_DIR}")


if __name__ == "__main__":
    main()
