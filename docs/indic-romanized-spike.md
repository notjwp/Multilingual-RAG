# Indic Romanized Spike — Hindi

**Status: COMPLETE.**

**Verdict: romanized input is unusable as-is (retention 0.08–0.12); transliterating romanized →
native script is the fix and recovers most of the gap (retention 0.75). Adopt a
transliterate-then-embed step in the query path. The transliteration *tool* still needs work
(IndicXlit won't install on Py3.13; the LLM works but is imperfect + costs credits).**

**The question:** can users query in **romanized** Hindi (Latin letters, as typed on a normal
keyboard), or must it be native Devanagari? And if romanized fails, does transliterating it back
to native script fix it?

---

## Method

| | |
|---|---|
| Index | 240 XQuAD-hi gold paragraphs + 20,000 MIRACL-hi distractors, all **native Devanagari** |
| Queries | 1,190 XQuAD-hi questions |
| Embedder | bge-m3 (the production model), fp16 |
| Search | exact brute-force cosine (measuring the query representation, not an ANN index) |
| Romanized | ISO transliteration with **diacritics stripped** — how people actually type (`भारत की राजधानी` → `bharata ki rajadhani`), and lossy on purpose so recovery is a real problem |
| Cost | $0 for A/B (local); condition C uses ~80 NIM calls |

**Tooling note:** the intended local transliterator, **IndicXlit** (`ai4bharat-transliteration`),
**would not install** on Python 3.13 — its `fairseq==0.12.2` dependency fails to build. Rather
than fight it, condition C measures the fix with the LLM we already have (NIM), on a sample. The
core finding (B) needs no transliteration tool.

---

## Finding 1 — romanized Hindi retrieval collapses (FINAL)

| condition | recall@1 | recall@5 | recall@10 | MRR | retention (r@5 / native) |
|---|---:|---:|---:|---:|---:|
| **native** (Devanagari) | 0.778 | **0.904** | 0.931 | 0.833 | — *(sanity gate ✅)* |
| **romanized-raw** | 0.047 | **0.107** | 0.143 | 0.074 | **0.118** |

**bge-m3 essentially cannot retrieve from romanized Hindi.** Recall@5 falls from **0.904 → 0.107**
against 20k distractors — it keeps only **12%** of native performance. The 6-doc probe earlier in
the session suggested ~0.6; at real scale the failure is far worse, because romanization strips
the language signal and 20k distractors punish that hard.

This decisively fails every "usable as-is" bar (0.118 ≪ the 0.60 floor). **Romanized input needs
a fix — it is not optional.**

Native at 0.904 passes the sanity gate (matches M0-era Hindi-adjacent numbers), so the harness is
sound and the romanized number is real, not an artifact.

---

## Finding 2 — transliteration recovers most of the gap (FINAL)

Three conditions on the **same 80 queries** (apples-to-apples):

| condition | recall@1 | recall@5 | recall@10 | MRR | retention (r@5 / native) |
|---|---:|---:|---:|---:|---:|
| **native** | 0.800 | 0.938 | 0.963 | 0.859 | — |
| **romanized-raw** | 0.050 | 0.075 | 0.113 | 0.060 | **0.080** |
| **transliterated → native** (NIM) | 0.450 | 0.700 | 0.738 | 0.535 | **0.747** |

Converting romanized → native before embedding lifts retention from **8% → 75%**. The mechanism
works: the problem was purely the *script/language signal*, and restoring native script restores
most of retrieval.

**Why 0.75 and not higher** — and this is the actionable part:

1. **The LLM transliterator is imperfect.** It's a telephone game: English proper nouns were
   transliterated into Devanagari (XQuAD), then romanized lossily, then recovered — errors
   accumulate (`taikala` → recovered as टाइम "time" instead of टैकल "tackle"; `saika` → साइकल
   "cycle" instead of सैक "sack"). A purpose-built transliterator (IndicXlit, properly installed)
   or a stronger model would recover more.
2. **XQuAD-hi is an adversarial corpus for this test** — it's NFL/Panthers content, dense with
   English names spelled in Devanagari, which are exactly what the romanize→recover round-trip
   mangles. On genuinely native Hindi text (fewer transliterated loanwords), transliteration
   should do **better** than 0.75. So treat 0.747 as a **floor**, not a ceiling.

---

## Decision (pre-registered thresholds)

| Threshold | Result |
|---|---|
| B (raw) retention ≥ 0.85 → no fix needed | ❌ **0.08–0.12** — romanized is unusable |
| C (transliterated) retention ≥ 0.85 → adopt transliteration | ➖ 0.747 — just below |
| both < 0.60 → transliteration insufficient | ❌ not this case |
| **C in 0.60–0.85 → partial: adopt transliteration, improve the transliterator** | ✅ **this** |

### Verdict — build it

**Adopt a query-path step: detect romanized input → transliterate to native script → embed.**
The approach is proven (0.08 → 0.75 retention). The open item is the *transliterator quality*:

- **Preferred:** get **IndicXlit** running (a Python-3.10 side environment or the AI4Bharat
  HTTP API) — purpose-built, local/free, no per-query cost. Measure it against this 0.747 LLM
  baseline; it should beat it.
- **Works today:** the **NIM LLM** path (what we measured) — but it adds a model call + credits +
  latency to every romanized query.
- **Consider combining** with hybrid BM25 or a reranker to claw back the last gap (partial-band
  guidance).

---

## Caveats (stated up front)

- **Clean romanization is a best case.** Diacritic-stripped ISO is more consistent than real
  human typing (which also drops schwas, varies spelling). So romanized-raw (B) may be a mild
  *over*-estimate — real input could be even worse — and the transliteration fix (C) is measured
  on relatively well-formed input.
- **Hindi only.** The romanized→native mechanism is script-agnostic (Brahmic), so this is a
  strong signal for Kannada/Telugu, but kn/te still need their own real romanized eval sets
  before shipping (kn has no corpus at all).

## What graduates / dies

- **Graduates:** the numbers, the decision, and the IndicXlit-won't-install note → `docs/`.
- **Dies:** everything in `spikes/indic/`.
