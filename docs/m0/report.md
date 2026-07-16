# M0 — Thesis Spike Report

**Status: COMPLETE.** All gates passed. Verdict below.

**The question:** does dense retrieval actually work *across languages* on this corpus? It is
the entire premise of this project and nothing had ever tested it.

**The answer: yes — with `BAAI/bge-m3`, and not with `multilingual-e5-large`.**

---

## Verdict

**Adopt `BAAI/bge-m3`. Dense-only cross-lingual retrieval is viable. No reranking milestone
is needed.**

Every bge-m3 retention ratio clears the pre-registered 0.85 bar (minimum **0.894**), landing
it in the *"crossing languages is nearly free"* band. e5 lands three of five cells in the
*"needs help"* band (minimum **0.667**).

The thesis holds. The project's premise is sound. The embedding model choice — which was
never measured before today — was the thing that decided it.

---

## Method

| | |
|---|---|
| Gold | XQuAD — 240 paragraphs, 1,190 questions, professionally translated and **parallel** across languages, so cross-lingual ground truth is exact |
| Distractors | 20,000 miracl-corpus Wikipedia passages per document language |
| Index | 20,240 passages per cell (240 gold + 20k distractors) |
| Search | **Exact brute-force cosine** (numpy), not ANN — we are measuring the embedding model, and HNSW would contribute its own recall loss |
| Languages | `en`, `es` (Latin control), `zh`, `th` (no-whitespace scripts) |
| Models | `intfloat/multilingual-e5-large`, `BAAI/bge-m3` — both free, local, 1024-dim, capped at 512 tokens here |
| Hardware | RTX 3050 Laptop, 4GB (~3.2GB free), fp16, batch 16 |
| Wall time | ~11 min per model to embed 85,720 texts |
| Cost | **$0** |

### Corpus integrity checks (all passed)

- **Parallel alignment verified.** Identical qid order and question→paragraph mapping across
  all four languages. `build_corpus.py` hard-fails otherwise, because every cross-lingual
  number depends on "paragraph *i* means the same thing in every language."
- **3 English gold paragraphs were found duplicated in the distractor pool and dropped.**
  XQuAD contexts come from Wikipedia via SQuAD and miracl-corpus `en` is also Wikipedia, so
  gold genuinely reappears as a distractor. Unchecked, three `en→en` queries would have had
  an identical twin in the index — corrupting the cell used as the harness sanity gate.
- **BOM stripped.** XQuAD's `es` and `th` contexts begin with a literal U+FEFF that would
  otherwise embed as a junk leading token.

---

## Finding 1 — the chunker silently destroys non-Latin documents

`ingestion/chunker.py` tokenizes on `re.compile(r"\S+")`. Chinese and Thai don't put spaces
between words, and their punctuation is also non-whitespace, so a whole article collapses
into **one** "token" window.

| lang | chars | real tokens | `\S+` chunks | correct chunks | **silently dropped by e5** |
|---|---:|---:|---:|---:|---:|
| zh | 10,151 | 7,015 | **1** | 16 | **92.7%** |
| zh | 20,580 | 13,486 | **1** | 31 | **96.2%** |
| zh | 10,666 | 6,743 | **1** | 16 | **92.4%** |
| th | 30,010 | 8,061 | 2 | 18 | **87.3%** |
| th | 11,745 | 2,784 | **1** | 7 | **81.6%** |
| th | 26,674 | 7,330 | 2 | 17 | **86.0%** |
| en | 72,232 | 17,259 | 15 | 39 | 55.5% |
| en | 50,598 | 10,956 | 11 | 25 | 48.6% |
| en | 17,548 | 4,217 | 4 | 10 | 51.4% |

**Two independent defects that compound:**

1. **`\S+` is the wrong tokenizer.** A 20,580-character Chinese article becomes a single
   chunk; e5 embeds its first 512 tokens and discards **96.2%**. No error is raised.
2. **`chunk_size_tokens=800` is in the wrong unit.** It counts *words*; 800 words is
   ~1,240–1,663 real BPE tokens, well past e5's 512 cap. This is why **even English loses
   ~50%** — a bug that fixing the tokenizer alone would not catch.

*Caveat, stated against interest:* for multi-chunk documents the "dropped" figure sums
`min(tokens, 512)` per chunk and so double-counts the 120-token overlap — the English losses
are if anything **understated**. The Chinese figures are exact (one chunk, no overlap).

**bge-m3's 8192-token limit mitigates but does not fix this**: less truncation, but one
vector per whole article still destroys retrieval granularity — you match documents, not
passages.

---

## Finding 2 — cross-lingual retrieval quality

### multilingual-e5-large

| cell | recall@1 | recall@5 | recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| `en→en` *(sanity)* | 0.870 | **0.949** | 0.969 | 0.906 | 0.921 |
| `es→es` | 0.854 | 0.942 | 0.961 | 0.892 | 0.908 |
| `zh→zh` | 0.824 | 0.932 | 0.950 | 0.867 | 0.887 |
| `th→th` | 0.792 | 0.932 | 0.948 | 0.850 | 0.874 |
| `en→es` | 0.834 | 0.934 | 0.952 | 0.875 | 0.894 |
| `es→en` | 0.774 | 0.890 | 0.919 | 0.824 | 0.847 |
| `en→zh` | **0.304** | 0.769 | 0.836 | 0.510 | 0.591 |
| `en→th` | 0.582 | 0.751 | 0.831 | 0.657 | 0.699 |
| `zh→en` | 0.455 | 0.633 | 0.695 | 0.532 | 0.571 |

### bge-m3

| cell | recall@1 | recall@5 | recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| `en→en` *(sanity)* | 0.801 | **0.920** | 0.949 | 0.855 | 0.878 |
| `es→es` | 0.800 | 0.913 | 0.941 | 0.849 | 0.872 |
| `zh→zh` | 0.798 | 0.930 | 0.948 | 0.854 | 0.877 |
| `th→th` | 0.743 | 0.897 | 0.929 | 0.810 | 0.839 |
| `en→es` | 0.779 | 0.897 | 0.927 | 0.831 | 0.855 |
| `es→en` | 0.755 | 0.886 | 0.926 | 0.815 | 0.842 |
| `en→zh` | 0.673 | 0.836 | 0.880 | 0.745 | 0.778 |
| `en→th` | 0.668 | 0.827 | 0.882 | 0.741 | 0.775 |
| `zh→en` | 0.645 | 0.823 | 0.870 | 0.724 | 0.759 |

### Retention ratio — the headline number

`retention = cross-lingual recall@5 ÷ monolingual recall@5` for the same document language.
Self-normalizing, so "Thai is intrinsically hard" stops being a confound and the number
isolates what crossing a language boundary actually costs.

| cell | e5 | **bge-m3** |
|---|---:|---:|
| `en→es` | 0.992 | 0.983 |
| `es→en` | 0.938 | **0.963** |
| `en→th` | 0.806 | **0.921** |
| `en→zh` | 0.825 | **0.899** |
| `zh→en` | **0.667** | **0.894** |
| **minimum** | **0.667** | **0.894** |

### Reading the result

**Neither model is bad at non-Latin scripts.** Monolingual `zh` is 0.932 (e5) and 0.930
(bge-m3); `th` is 0.932 and 0.897. Both clear the 0.60 absolute floor with enormous room.
The scripts were never the problem.

**The difference is entirely in *crossing*.** e5 degrades sharply when the query language
differs from the document language and the scripts differ; bge-m3 barely degrades at all.
The `zh→en` cell is the clearest: e5 **0.633** vs bge-m3 **0.823**.

**e5 is marginally better monolingually** (`en` 0.949 vs 0.920, `th` 0.932 vs 0.897) — and
that is exactly the trade this project should refuse. Monolingual quality is not the product;
crossing languages is.

**The recall@1 column explains why no reranker is needed.** e5's `en→zh` recall@1 is 0.304
against recall@5 of 0.769 — it *finds* the right document but *ranks* it badly, which is the
classic signal that a cross-encoder reranker would help. bge-m3 posts **0.673** recall@1 on
the same cell. It ranks correctly on its own, so the reranking milestone that e5 would have
forced is unnecessary.

**Asymmetry worth noting:** for both models, `en→zh` beats `zh→en`. Asking in English about
Chinese documents works better than the reverse.

---

## Verification gates

Criteria recorded **before** any results existed.

| Gate | Requirement | Result |
|---|---|---|
| **Alignment** | XQuAD parallel structure intact | ✅ passed |
| **Dedup** | No gold paragraph duplicated in distractors | ✅ passed (3 dropped) |
| **Sanity** | `en→en` recall@5 ≈ 0.9+ | ✅ passed — e5 **0.949**, bge-m3 **0.920** |
| **Prefix probe** | e5 must beat e5-without-prefixes, proving prefixes are applied | ✅ passed — see below |

### The prefix gate, in detail (and a prediction I got wrong)

|  | recall@5 | hits |
|---|---:|---:|
| e5 **with** prefixes | 0.9487 | 1129/1190 |
| e5 **without** prefixes | 0.9403 | 1119/1190 |
| difference | **+0.0084** | |

McNemar on discordant pairs: **17 vs 7**, exact binomial **p = 0.064** — directionally
correct, but *not* significant at 0.05.

**The gate passes, on decisive evidence that is not the score gap.** The two runs produced
embeddings that are **not identical** (`array_equal = False`; query[0] cosine = **0.9829**).
Had the prefixes never been applied, the arrays would have been bit-identical. They aren't.
So `"query: "`/`"passage: "` definitively reach the model, e5 was configured correctly, and
its numbers are valid — which means **bge-m3's win is a fair win, not an artifact of a
misconfigured opponent.** That was the gate's actual job.

**I over-weighted this risk.** The plan called the prefix asymmetry "the single most likely
way this spike lies to you." On `en→en` it is worth **0.8 percentage points** and does not
reach significance. The trap was real and cheap to guard, but it was not the landmine
claimed.

*Caveat:* only `en→en` was probed — the easiest cell. Prefixes may matter more in harder
cross-lingual cells. This does not affect the ranking either way: e5's main run *had*
prefixes throughout and still lost.

---

## Decision thresholds

**Pre-registered before any results existed** — deciding what counts as success after seeing
the numbers is how a spike gets talked into whatever you already wanted.

| Retention ratio | Verdict | e5 | bge-m3 |
|---|---|---|---|
| **≥ 0.85** | Crossing languages is nearly free. Dense-only viable. | 2 of 5 cells | **5 of 5 cells** ✅ |
| **0.60 – 0.85** | Viable but needs reranking and/or hybrid BM25. | **3 of 5 cells** | 0 |
| **< 0.60** | Dense-only fails. Query translation, or reconsider the model. | 0 | 0 |

**Absolute floor** (monolingual `zh`/`th` recall@5 ≥ 0.60): both models pass comfortably.

**Tie-breaker (prefer bge-m3 if close): not needed.** bge-m3 wins on the merits.

---

## Why bge-m3 wins

| | e5-large | **bge-m3** |
|---|---|---|
| Cross-lingual retention (min) | 0.667 | **0.894** |
| Monolingual quality | *marginally better* | slightly lower |
| Max sequence | 512 | **8192** |
| Prefixes | required | **none** |
| Hybrid retrieval | — | **sparse + multi-vector, free** |
| Reranker needed? | **yes** (recall@1 0.304 on `en→zh`) | **no** (0.673) |

bge-m3 wins the only axis that defines this product, and throws in a 16× context window, a
simpler adapter, and free hybrid retrieval if ever needed. e5 wins only on monolingual
quality, which is not what is being built.

---

## Production implications

1. **Adopt bge-m3 as the `EmbeddingProvider`.** No prefixes; 1024-dim; 8192 context.
2. **`chunk_size_tokens=800` is legal under bge-m3** (8192 cap) but was *silently
   catastrophic* under e5's 512. Keep 800 — but only because bge-m3 won. **The model dictates
   this number**, which is the real lesson.
3. **The chunker must take a tokenizer as a dependency.** `\S+` must go; tiktoken would also
   be wrong (bge-m3 uses XLM-R's tokenizer). Tokenizer and max chunk size are properties
   **of the embedding model**, not global config. This fits the existing `Protocol` ports
   pattern: `EmbeddingProvider` exposes its tokenizer and limit; the chunker consumes them.
4. **No reranking milestone.** bge-m3's recall@1 makes it unnecessary — a milestone avoided
   on evidence, not opinion.
5. **Prefix handling still belongs inside the adapter**, should a future model need it.
6. **Zero API cost for embeddings.** ~11 min on a 3050 to embed 85,720 texts. Generation
   remains the only piece needing a free-tier API.

## What graduated

| Artifact | Home |
|---|---|
| Gold paragraphs + queries (4 langs, ~1.8 MB) | `data/eval/xquad/` — committed |
| Corpus regenerator (pinned + seeded) | `scripts/build_eval_corpus.py` |
| This report + raw result JSONs | `docs/m0/` |
| Distractors (~51 MB) | Not committed — regenerate with the script |

**`spikes/m0/` was deleted.** It was a measurement instrument, not a foundation.

### These numbers are a reference point, not a regression target

An earlier draft called the tables above "M3's regression floor." That was wrong, and acting
on it would cause a false alarm the first time M3 runs.

M0 measured **paragraph-level retrieval over a clean 20k index with exact brute-force
cosine**. M3 will measure the **full production pipeline** — real chunking, Chroma, an ANN
index, a different corpus construction. M3 scoring lower than 0.894 retention is *expected*
and does not by itself indicate a regression: it is a different measurement of a different
system.

What M0 licenses is narrower and still useful: *bge-m3 can sustain ≥0.894 cross-lingual
retention under ideal conditions.* If M3 comes in far below that, the gap is attributable to
the pipeline (chunking, ANN recall, retrieval config) rather than to the embedding model —
which is exactly the diagnostic separation worth having.

## Reproducibility

### Pinned revisions

Everything measured here used exactly these:

| Artifact | Revision | Status |
|---|---|---|
| `google/xquad` | `51adfef1c1287aab1d2d91b5bead9bcfb9c68583` | **Load-bearing** — `scripts/build_eval_corpus.py` pins it |
| `miracl/miracl-corpus` | `d921ec7e349ce0d28daf30b2da9da5ee698bef0d` | **Load-bearing** — `scripts/build_eval_corpus.py` pins it |
| `intfloat/multilingual-e5-large` | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` | Record — the losing arm |
| `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | **Pin this in the production adapter.** It is now the model the retrieval stack depends on. |

`SEED = 42` alone is **not** sufficient for reproducibility: `rng.sample()` is deterministic
only if the pool it samples from is byte-identical, which requires the pinned shard.

### Other controls

Exact brute-force search (no ANN nondeterminism); fp16 embeddings; both models capped at 512
tokens for a like-for-like comparison — bge-m3's 8192 advantage is measured in Finding 1, not
Finding 2, so this comparison does not flatter it.

### What is and isn't reproducible

The corpus is: `gold_*.jsonl` and `queries_*.jsonl` are committed under `data/eval/xquad/`,
and `python scripts/build_eval_corpus.py` regenerates the 51 MB of distractors byte-identically
from the pinned shard plus seed.

The measurement harness is **not**: `embed.py` and `run_matrix.py` were deleted with the spike,
a deliberate trade so the spike could not calcify into the system. M0 remains *re-derivable*
(~200 lines against byte-identical inputs), not re-runnable. The numbers survive in this
report and the raw JSONs beside it.
