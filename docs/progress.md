# Progress

Working reference for where this project stands. Updated as work lands.

**Last updated:** 2026-07-20
**Current decision:** Fix-in-place (Phases A–D). Rebuild rejected on evidence.
**Status:** Phases A, B, C, C4, D all ✅. Romanized query-path ✅ — Hindi (default) + opt-in
Kannada/Telugu via a **local multi-class MuRIL detector** (no per-query network) or the google
detector.
**Next candidates (none started):** MuRIL-for-retrieval; the two Phase-D deferrals (Chroma server
mode, torch-image slimming).

---

## Timeline

| # | Stage | Status |
|---|---|---|
| — | Codebase audit | ✅ done |
| M0 | Cross-lingual thesis spike | ✅ done — verdict: adopt **bge-m3** |
| — | M0 close-out (pin/graduate/delete spike) | ✅ done |
| — | Fix-vs-rebuild decision | ✅ **fix-in-place** |
| A | Security (P0) | ✅ **done** — leak closed, 55 tests green |
| B | Make it measurable | ✅ **done** — real citations + free live retrieval eval |
| C | Multilingual correctness (C1–C3) | ✅ **done** — CJK chunking fixed, bge-m3 in prod, no regression (0.902) |
| C4 | Free generation (OpenAI-compatible) | ✅ **DONE** — live NIM query verified, zero OpenAI calls |
| — | Indic romanized spike (Hindi) | ✅ done — romanized collapses (0.08 retention); transliterate→native recovers to 0.75. **Build it.** See docs/indic-romanized-spike.md |
| D | Correctness + light scale | ✅ **DONE** — broken DELETE fixed (live), dedup, DB tests, threadpool, no regression |
| — | Indic romanized query-path (Hindi) | ✅ **DONE** — detect→transliterate; romanized recall 0.20→0.67 (3.3×), English untouched, live-verified |
| — | Kannada/Telugu (detect-the-language) | ✅ **DONE** — google `detect()` routes per-language; validated kn 0.963, te 0.925, 0% English FP; opt-in |
| — | Local multi-class MuRIL detector | ✅ **DONE** — frozen MuRIL + multinomial LR (hi/kn/te/other); held-out hi 1.0 / kn 0.987 / te 0.920, 0 FP; shipped kn 0.963 / te 0.975, no per-query network |

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

### Phase A — Security (P0) ✅ DONE
- ✅ A1 auth on `/v1/query` (`api/routes/query.py`)
- ✅ A2 `user_id` in Chroma metadata + protocol `search`/`upsert`/`delete` signatures;
  server-side `$and` scope; **storage ids namespaced by user** so identical cross-user uploads
  don't overwrite (the non-obvious interaction — see plan)
- ✅ A3 threaded `user_id` retrieval → query service; all 5 call sites fixed (mypy-enforced)
- ✅ A4 refuse default `jwt_secret_key` in prod (`core/config.py`, `model_validator`) — verified
- ✅ A5 cap uploads (`Settings.max_upload_bytes` + capped read → 413)
- ✅ tests: 401, cross-tenant isolation (real Chroma), no-overwrite, scoped delete, rejected
  `user_id` filter — **55 passed · ruff ✅ · mypy ✅**
- ⚠️ Re-index required: pre-existing vectors lack `user_id` → invisible (fails closed). README
  updated. No local `data/chroma` existed, so nothing to wipe here.

### Phase B — Measurable ✅ DONE (zero-spend)
Reshaped by the "never spend on OpenAI" constraint: retrieval eval runs **free** on local
bge-m3; generation-side metric *values* deferred to C (need a free LLM).
- ✅ B1 real `[n]` citation parsing (`generation/citations.py`) — generator cites only marked
  chunks, never everything; `parse_cited_results` reused by the free generator in C
- ✅ B2 `citation_precision`/`citation_recall` (`metrics.py`); `FaithfulnessJudge` Protocol +
  `average_faithfulness` (`evaluation/faithfulness.py`) — impl in C
- ✅ B3 `BgeM3EmbeddingProvider` (`embeddings/bge_embeddings.py`), no prefixes, pinned revision,
  `lru_cache` load; `sentence-transformers` under optional `[eval]` extra (core stays torch-free)
- ✅ B4 live harness (`evaluation/harness.py`) — maps XQuAD to one-chunk docs, ingests +
  retrieves under `user_id="__eval__"` in an isolated Chroma; provider-agnostic (fake-tested)
- ✅ B5 `load_xquad_corpus` (`datasets.py`, fixture mode untouched); `run.py --live/--langs/--sample`
- ✅ 64 passed + 1 skipped (bge-m3 test opt-in via `RUN_MODEL_TESTS=1`); ruff + mypy clean
- **Deferred to C:** real generation numbers (citation precision/faithfulness values, answer
  language), production embedding swap + Chroma re-index, free generation adapter
- Decision noted: `citation_precision` is 1.0 when nothing is cited (vacuous) — always paired
  with recall (0.0 there), so a non-citing answer is still penalised

**Free live retrieval baseline** (the deliverable): `python -m multilingual_rag.evaluation.run
--live --langs en zh --k 5` (needs `pip install -e ".[eval]"`).

Recorded baseline (en+zh, 40,480 docs, 2,380 queries, $0):
**recall@5 = 0.903 · MRR = 0.815 · nDCG@5 = 0.837.**

Wiring confirmed vs M0 (bge-m3 monolingual, exact search: en 0.920 / zh 0.930, avg ~0.925). The
0.903 is close-but-lower as predicted — explained by (1) a single mixed en+zh index = 40k
distractors/query vs M0's per-language 20k, and (2) Chroma HNSW (approximate) vs M0's exact
search. The real pipeline reproduces the spike's retrieval quality. `language_match_rate=0.0`
is expected (no generation until C). Note: combined vs per-language, so not perfectly
apples-to-apples; an en-only/zh-only run would isolate it if ever needed.

### Phase C — Multilingual correctness (C1–C3) ✅ code complete, offline
Scoped to C1–C3 (fully offline, no signup/spend); C4 (generation) is its own next phase.
- ✅ C1 tokenizer-aware chunking: `Tokenizer` protocol + `BgeM3Tokenizer`
  (`ingestion/tokenizer.py`, loads only the tokenizer, not the 2.2 GB model); `TextChunker`
  windows over token ids. **Proven end-to-end:** a long Chinese doc that `\S+` collapsed to 1
  blob now yields **9 correctly-sized chunks** through the real `IngestionService`.
- ✅ C2 `resolve_answer_language` (`generation/language.py`) wired into the generator — a short
  query that detects as `"unknown"` now answers in the evidence's language, never "unknown".
- ✅ C3 `build_embedding_provider` factory (`embeddings/factory.py`); `embedding_provider` /
  `embedding_device` settings; wired into query route + ingestion job; `sentence-transformers`
  moved to **core** deps (torch in the image — Phase D slims it). Per-request safe via the
  model's `@lru_cache`.
- ✅ 73 passed + 2 skipped (model tests opt-in), ruff + mypy clean.
- ✅ No-regression eval passed: recall@5 = **0.9021** (Phase B was 0.9034 — identical within
  HNSW noise). Confirms the production bge-m3 embedding path works after the 1024-dim re-index.

**Findings to carry into C4:**
- langdetect returns BCP-47-ish codes (`zh-cn`/`zh-tw`), **not** ISO `zh`. The M0 eval corpus
  labels languages as `zh`/`en`/`es`/`th`, so `language_match_rate` will read 0 even when the
  answer language is correct unless codes are normalized. Normalize (e.g. `zh-cn` → `zh`) when
  wiring generation eval in C4.

### Phase C4 — Free generation ✅ code complete, ⏳ live verification blocked on an API key
- ✅ **`OpenAICompatibleAnswerGenerator`** (`generation/openai_compatible_generator.py`) — one
  adapter for *any* OpenAI-compatible `chat.completions` endpoint. Reuses `build_answer_prompt`,
  `parse_cited_results`, `resolve_answer_language`. 429 → `generation_rate_limited`; a vanished
  model → `generation_model_unavailable` **naming `GENERATION_MODEL`** (catalogs rotate).
- ✅ **The provider is a URL, not a code path.** `GENERATION_BASE_URL` selects NVIDIA NIM
  (default), OpenRouter, Groq, a local Ollama/vLLM shim, or OpenAI — **zero code change**.
  Tested: `test_provider_is_just_a_url_not_a_code_path`.
- ✅ **Env-driven:** `GENERATION_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL`.
  `.env.example` ships NIM as default with the alternatives as commented one-liners.
- ✅ **Collapsed two adapters + an enum into one.** Deleted `generation/openai_generator.py`
  (Responses API — redundant, OpenAI is reachable via `base_url`) and `generation/factory.py`
  (a factory with one option is noise). Dropped the incoherent
  `generation_provider: ["openai-compatible","openai"]` enum — OpenAI *is* OpenAI-compatible.
- ✅ Boot guard: production/staging refuse to start without `GENERATION_API_KEY`
  (local/test stay permissive so the suite runs — generator raises a named `AppError` if used).
- ✅ `LlmFaithfulnessJudge` (`evaluation/llm_judge.py`) implements B's Protocol.
- ✅ `normalize_language_code` — **kills the `zh-cn` vs `zh` false-zero** in
  `language_match_rate` (normalized on both sides, inside the metric so it can't be bypassed).
- ✅ Harness generation eval: optional generator + judge, **sampled** (`--gen-sample`, default
  50) so a free-tier quota survives; retrieval still covers the full corpus. Generation metrics
  score **only generated examples** — otherwise a sampled run would look broken.
- ✅ `run.py --generate / --judge / --gen-sample`. Retrieval-only path verified unchanged.
- ✅ 80 passed + 2 skipped; ruff + mypy clean. `grep -ri openrouter src/` → only provider-example
  docs; no vendor name in any identifier or config key.
- ✅ **Live-verified** against NVIDIA NIM: generation returns a grounded, `[1]`-cited answer
  with **zero OpenAI calls**. The whole `/v1/query` path is now free.
- ✅ **Added a request timeout** (`generation_timeout_seconds`, default 60) after a live
  incident: `meta/llama-3.3-70b-instruct` **hung** on NIM's free tier (cold/overloaded), and the
  SDK's 600s default would have blocked the request for 10 min. Now → 504 `generation_timeout`
  with an actionable message. Default model switched to **`meta/llama-3.1-8b-instruct`** (verified
  responds in ~1.5s). Diagnosis: the 8B answered instantly through the same path, proving key +
  network + adapter were all fine — only the 70B was unresponsive.
- ✅ **Full feature test done** (live, this session):
  - 82/82 tests pass with `RUN_MODEL_TESTS=1` (incl. real bge-m3 embed + CJK chunker).
  - **Indic generation works**: NIM Llama-3.1-8B answers grounded + cited in **Hindi, Kannada,
    Telugu native script** (and zh) — C2 language resolution fills in from evidence for the
    short "unknown" queries.
  - End-to-end eval `--live --generate --judge`: recall@5 0.995 (200-doc sample), citation
    precision 0.8 / recall 0.6, faithfulness 1.0, language_match 1.0 (non-zero ✓), generation
    sampled 5/200.
- ✅ **Full-stack HTTP path verified live** (Postgres+Redis+uvicorn+Celery worker+bge-m3+Chroma
  +NIM): signup→JWT; 401 without token; reserved-filter→400; **cross-tenant isolation** (2nd
  user sees 0 of the 1st's docs); async upload→Celery ingest→succeeded; a Chinese doc ingested
  as **2 chunks** (C1 fix through the real pipeline); Chinese query returned a grounded, cited
  answer **in Chinese with zero OpenAI calls**; cross-lingual en-query→zh-doc retrieved. All
  passed.
- 🐛 **Fixed a real bug found during the run:** Alembic migrations required `psycopg2` (env.py
  rewrites the async URL to a sync one) but it was never a dependency — so migrations had never
  worked against Postgres. Added `psycopg2-binary` to `pyproject.toml`. (This is exactly the
  "DB layer never exercised" gap the audit flagged; D9 covers real DB tests.)
- ⚠️ NIM free tier is **credit-based** (1000 credits, 40 RPM) — finite. Hence `--gen-sample 20`.
- ⚠️ Free-tier large models can hang; keep a small responsive model as the default and treat
  the model id as an env knob (`GENERATION_MODEL`).

### Phase D — Correctness + light scale ✅ DONE
Scoped to correctness + cheap robustness (Chroma server mode + torch-image slimming deferred as
over-engineering for one machine). Sequenced tests-first so they exposed the bugs before fixing.
- ✅ D9 DB-layer tests against **real Postgres** (`tests/integration/test_db_layer.py`, gated —
  skip if Postgres unreachable; added `pytest-asyncio`). The coverage that was missing; wrote
  bug-pinning tests RED, then fixed to green. Also the first-ever `run_ingestion_job` test.
- ✅ D4 broken `DELETE` fixed — `ondelete="CASCADE"` on child FKs (`SET NULL` on
  `ingestion_jobs.document_id`) in `db/models.py` + migration `0002`. **Verified live:**
  `DELETE /v1/documents/{id}` now returns 200 (was `IntegrityError`).
- ✅ D7 file checksum = content hash (`repository.py`), not the path string.
- ✅ D6 content-addressed **and user-scoped** dedup: `document_id = uuid5(f"{user_id}:{checksum}")`
  (was mixing a per-upload uuid4 path in) + `(user_id, checksum)` unique constraint (migration
  `0003`). **Verified live:** two identical uploads → one document, same id. (Refinement of the
  approved plan: user-scoping avoids a cross-tenant clobber that checksum-alone would introduce.)
- ✅ D5 dual-write compensation in `run_ingestion_job` — DB rows then vectors then one commit;
  best-effort vector cleanup on failure; refactored to inject deps for testing. The D5 test caught
  a real bug in the fix (reading a rolled-back ORM object silently no-op'd cleanup → capture
  fields into locals first).
- ✅ D8 legacy `DocumentStore` / `DocumentIndexingService` / `document_store_path` deleted.
- ✅ D1 offload the blocking RAG core to a threadpool (`await asyncio.to_thread(...)` in the query
  route) — not an async rewrite (local models + sync client can't be truly async).
- ✅ D3 memoize the query service on `app.state` (built once, not per request; lazy so the offline
  suite never loads the 2.2 GB model at startup).
- ✅ Gates green throughout; 87 passed with `RUN_MODEL_TESTS=1`. Eval unchanged (recall@5 0.995 on
  the sample — retrieval untouched). **Deferred:** Chroma server mode, torch-image slimming.

### Indic romanized query-path — Hindi ✅ DONE
Users can type Hindi in the Latin alphabet (`bharat ki rajdhani kya hai`) and hit the
native-Devanagari index. See `docs/indic-romanized-spike.md` for the motivating spike.
- ✅ `Transliterator` **port** + adapters (`transliteration/`): **google** (default — googletrans,
  best quality, free, network per query, local rule-based fallback baked in), **indicxlit** (local
  offline neural, `psidharth567/indic-xlit-50M`, pinned; feasibility-proven on Py3.13),
  **rule-based** (`indic-transliteration`), **llm** (reuses the generation endpoint). Env-driven
  via `TRANSLITERATION_PROVIDER`.
- ✅ **Design pivoted on eval evidence.** First built dual-query (search raw + transliterated,
  fuse). The eval showed *every* fusion (max-cosine, RRF, confidence-routing) dragged Hindi recall
  *below* pure transliteration (~0.56 vs ~0.67) — the raw search's noise is unavoidable when you
  can't tell which form is right. Switched to **detection** (`transliteration/detect.py`): a cheap
  distinctly-Hindi function-word check (`kya`/`hai`/`kaun`/`nahi`, English collisions excluded)
  decides *whether* to transliterate. Detected → search the transliterated form; plain English →
  raw, untouched.
- ✅ Measured (XQuAD-hi, 10k distractors, 150 queries, google, recall@5): native **0.947** ·
  romanized-raw **0.204** · transliterated **0.676** · **shipped 0.669**. Romanized recovers
  **0.20 → 0.67 (3.3×)**; detection recall **98.7%**, so shipped ≈ the transliterated ceiling;
  **0 English false positives**. (Below the spike's 0.747 LLM figure — that was a different corpus
  cut; the ceiling here is google's transliteration on English-name-heavy XQuAD-hi.)
- ✅ Eval tooling: `hi` added to `build_eval_corpus.py`; `scripts/eval_romanized.py` (romanizer +
  4 conditions). Live-verified through the real `RetrievalService` (bge-m3 + Chroma + google):
  romanized→correct doc, English not transliterated (and still cross-lingual). 109 passed, gates green.
- ✅ **Opt-in MuRIL detector** (`TRANSLITERATION_DETECTOR=muril`): a frozen `google/muril-base-cased`
  feature extractor (`transliteration/muril.py`, pinned) + a LogisticRegression head
  (`scripts/train_romanized_detector.py`, artifact `data/models/romanized_indic_detector.joblib`,
  KB-sized, committed). Started as a binary Hindi head; **now multinomial (hi/kn/te/other)** — see the
  local-multi-class section below. Default stays **word-list** (fast, no model, hermetic tests); MuRIL
  loads lazily on CPU only when opted in, with word-list fallback on any failure.
- **Deferred:** full MuRIL fine-tune; MuRIL for retrieval (the actual ~0.67 ceiling).

### Kannada/Telugu — detect-the-language ✅ DONE (opt-in)
Extends the romanized path to kn/te. googletrans already transliterates them; the gaps were
detection (hi-only) and routing (fixed to `languages[0]`).
- ✅ `detect.py`: `detect_target_language(...) -> str|None` (bool `is_romanized_indic` kept as a
  wrapper). New **`google` detector** (`TRANSLITERATION_DETECTOR=google`) uses googletrans
  `detect()` to identify hi/kn/te — the only path that supports Kannada/Telugu without per-language
  training data — with the Hindi word-list as a safety net on failure.
- ✅ `service.py` routes transliteration to the **detected** language (not `languages[0]`);
  `TRANSLITERATION_LANGUAGES=hi,kn,te` (comma-parsed) enables them. Transliterators were already
  kn/te-ready.
- ✅ Validation: no ready romanized kn/te corpus (IndicQA-romanized lacks them, FLORES-plus gated,
  script-based sets unloadable), so `scripts/build_indic_romanized_eval.py` synthesizes one from
  **Wikipedia** (native kn/te sentences) + the `indic_transliteration` romanizer;
  `eval_romanized.py` generalized to `--lang`. **Measured (self-retrieval, 1800 docs, 80 q):**
  romanized-raw→transliterated **kn 0.588→0.975, te 0.588→1.000**; shipped **kn 0.963 / te 0.925**
  (PASS); google detection 93.8% / 87.5%; **0/40 English false-positives**. (Easier eval than hi's
  XQuAD — proves the mechanism, not a kn/te-beats-hi claim.)
- ✅ Default unchanged (hi, word-list, no network); kn/te are opt-in. 101 unit tests green
  (googletrans mocked → hermetic).

### Local multi-class MuRIL detector ✅ DONE (opt-in)
Removes the google detector's per-query network call for kn/te: the MuRIL head went from binary
(is-hi) to **multinomial (hi/kn/te/other)**, so `TRANSLITERATION_DETECTOR=muril` now detects all
three languages *locally*.
- ✅ `train_romanized_detector.py` retrained multi-class: hi (XQuAD-hi), kn/te (Wikipedia
  *distractors* romanized — gold held out so the retrieval eval stays honest), other (en+es);
  multinomial LR on frozen MuRIL features → `data/models/romanized_indic_detector.joblib`
  (replaces the binary `romanized_hi_detector.joblib`). `detect_target_language(..., detector="muril")`
  now returns hi/kn/te; `_MurilDetector.predict_language`.
- ✅ **A debugging catch:** the first run showed kn 0.000 / hi 0.525 — a threshold bug (a 0.5
  max-proba floor drops correct predictions, whose max-proba is naturally <0.5 in a 4-class problem),
  *not* the model. The explicit "other" class already gives 0 leakage, so trust argmax (threshold 0).
  A diagnostic (char-ngram vs MuRIL) confirmed MuRIL separates the languages fine.
- ✅ **Held-out:** hi 1.000 / kn 0.987 / te 0.920, **0 other→Indic FP**. **Shipped (local detection,
  no network):** kn 0.963 / te 0.975 (PASS), detection 95% / 90%, 0/40 English FP — matching/beating
  the google detector. Note: char n-grams are an even-lighter alternative (no ~950 MB model) that
  scored comparably in the diagnostic; MuRIL was the requested approach.
- ✅ Default still word-list/hi (no model); muril is opt-in, lazy CPU, word-list fallback.

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
