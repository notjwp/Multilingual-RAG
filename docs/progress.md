# Progress

Working reference for where this project stands. Updated as work lands.

**Last updated:** 2026-07-16
**Current decision:** Fix-in-place (Phases A–D). Rebuild rejected on evidence.
**Next action:** Phase A (security P0) — not yet started.

---

## Timeline

| # | Stage | Status |
|---|---|---|
| — | Codebase audit | ✅ done |
| M0 | Cross-lingual thesis spike | ✅ done — verdict: adopt **bge-m3** |
| — | M0 close-out (pin/graduate/delete spike) | ✅ done |
| — | Fix-vs-rebuild decision | ✅ **fix-in-place** |
| A | Security (P0) | ⬜ not started |
| B | Make it measurable | ⬜ not started |
| C | Multilingual correctness + bge-m3 | ⬜ not started |
| D | Async + infra | ⬜ not started |

---

## Done

### Audit
Full read of 55 source files. Found ~10 localized defects (catalogued in
`architecture.md §7`). Conclusion: architecture is sound (ports/adapters, frozen models, strict
mypy, 51 passing tests); the roadmap was ordered by layer instead of risk, and there is a live
cross-tenant data leak.

### M0 — thesis spike ✅
**Question:** does dense retrieval work cross-lingually on this corpus? Never tested before.
**Answer:** yes, with `bge-m3`; not adequately with `multilingual-e5-large`.

- Corpus: XQuAD gold (parallel, exact cross-lingual ground truth) + 20k MIRACL distractors/lang,
  4 languages (en/es/zh/th).
- bge-m3 retention ≥ **0.894** on every cross-lingual cell; e5 floor **0.667** (3 cells in the
  "needs help" band). bge-m3's recall@1 also removes the need for a reranker milestone.
- Chunker defect quantified: `\S+` drops **96%** of a Chinese article; `chunk_size_tokens=800`
  (wrong unit) drops **~50%** of English.
- Cost: **$0**, ~2h GPU (RTX 3050).
- Full report: `docs/m0/report.md`.

### M0 close-out ✅
- Both dataset revisions pinned in the regenerator (SEED alone isn't reproducible).
- Corpus graduated: `data/eval/xquad/` (gold+queries committed, ~1.8MB; distractors
  regenerable, gitignored).
- Regenerator moved to `scripts/build_eval_corpus.py`; verified **byte-identical** across two
  runs.
- Report → `docs/m0/report.md`; `spikes/` deleted (430MB reclaimed).
- Corrected two of my own claims in the report (4 pins not 2; "regression floor" → "reference
  point").

---

## Remaining — Fix-in-Place (full plan: `~/.claude/plans/purrfect-bubbling-eich.md`)

Legend: ⬜ todo · 🟡 in progress · ✅ done

### Pre-flight (clear the baseline) ✅
- ✅ Fixed ruff error — `tests/integration/test_documents_route.py:7` (import wrapped)
- ✅ Fixed mypy error — `workers/celery_app.py:22` (ignore code was wrong: `[misc]` →
  `[untyped-decorator]`, which was also masking the real error)
- Baseline now fully green: ruff ✅ · mypy ✅ (55 files) · pytest ✅ (51 passed)

### Phase A — Security (P0, ~1 day)
- ⬜ A1 auth on `/v1/query` (`api/routes/query.py`)
- ⬜ A2 `user_id` in Chroma metadata + `search` signature; server-side `$and` scope
- ⬜ A3 thread `user_id` retrieval → query service
- ⬜ A4 refuse default `jwt_secret_key` in prod (`core/config.py`)
- ⬜ A5 cap uploads (`Settings` + `save_upload_bytes`)
- ⬜ tests: 401, cross-tenant isolation, rejected `user_id` filter

### Phase B — Measurable (~4 days)
- ⬜ B1 parse real `[n]` citations (`openai_generator.py`)
- ⬜ B2 real eval harness through the live pipeline (`evaluation/harness.py`)
- ⬜ B3 citation precision/recall + faithfulness (`FaithfulnessJudge` port)
- ⬜ B4/B5 dataset schema + `--live` flag; leave legacy fixture alone

### Phase C — Multilingual + bge-m3 (~3 days)
- ⬜ C1 tokenizer-aware chunking (tokenizer as `EmbeddingProvider` dependency)
- ⬜ C2 fix `"unknown"`-language-into-prompt (`language.py` / generator)
- ⬜ C3 `BgeM3EmbeddingProvider` — **must not load per request** (couples with D3)
- ⬜ C4 generation → free-tier API (OpenRouter Qwen2.5-72B)
- ⬜ re-index Chroma (covers both `user_id` and 1024-dim change)

### Phase D — Async + infra (~1 week)
- ⬜ D1 async core (ports + services + adapters)
- ⬜ D2 Chroma server mode (`AsyncHttpClient` + compose service)
- ⬜ D3 lifespan singletons (`app.state`)
- ⬜ D4 fix broken `DELETE` — FK `ondelete=CASCADE` + migration
- ⬜ D5 DB-first dual-write with compensation
- ⬜ D6 content-addressed dedup + `(user_id, checksum)` constraint
- ⬜ D7 fix bogus file checksum (`repository.py:70`)
- ⬜ D8 delete legacy `DocumentStore` path
- ⬜ D9 DB-layer tests against real Postgres

---

## Standing invariants (don't regress)

- All three gates green each phase: `pytest` · `ruff check .` · `mypy src`.
- Phase B's eval harness becomes the regression guard for C and D — a phase that moves metrics
  down doesn't land.
- One re-index only, scheduled after Phase C.
- M3-vs-M0 caveat: the full pipeline will score below M0's 0.894 for legitimate reasons; that
  is not a regression.

## Open questions / deferred

- Generation provider final choice (OpenRouter vs Groq) — Groq's free Llama omits Chinese;
  leaning OpenRouter Qwen2.5-72B. Decide at C4.
- Free-tier rate limits constrain the faithfulness judge → sample, don't judge all 1190.
- Japanese not covered by the M0 corpus (XQuAD has no `ja`); revisit if needed.
- Hybrid BM25 + reranking deferred (bge-m3's dense recall made it unnecessary for now).
