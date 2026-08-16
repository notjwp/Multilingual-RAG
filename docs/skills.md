# Skills & Knowledge Map

The technical domains needed to work productively in this repo, each with the repo-specific
gotcha that generic tutorials won't teach you. This complements — does not repeat —
`CLAUDE.md` (terse conventions) and `docs/architecture.md` (structure).

Rated by how much this codebase leans on each: ●●● core, ●● frequent, ● occasional.

---

## ●●● Python typing under `mypy --strict`

Every function is fully annotated; `mypy src` must pass with zero errors. Untyped third-party
libs (celery, chromadb) need `# type: ignore[<specific-code>]` — the bare form is rejected.

- **Protocols** (`typing.Protocol`) are the backbone — structural typing lets a fake satisfy a
  port without inheritance. See `embeddings/base.py`, `vectorstores/base.py`,
  `generation/base.py`.
- Pydantic v2 `BaseModel` with `ConfigDict(frozen=True)`; collections are `tuple[...]`.
- `cast(...)` appears at trust boundaries (`request.app.state`, Chroma's untyped returns).

**Learn if unfamiliar:** structural vs nominal typing, `Protocol`, `cast`, `# type: ignore`
error codes.

## ●●● FastAPI — but not the idiomatic parts

The unusual bits matter more than the framework basics here.

- **DI via `app.state`, not `dependency_overrides`.** Read `docs/architecture.md §2.2` before
  adding any route. The pattern: declare a `Protocol`, add a `get_*_service` fallback helper,
  hang overrides off `app.state`.
- **Errors are `AppError`, never `HTTPException`** — one handler in `api/app.py` renders them.
- App is built by a `create_app(settings)` **factory**; tests construct their own.
- `Request.app.state.settings` is how routes reach config.

**Learn if unfamiliar:** FastAPI `APIRouter`, dependency injection, `TestClient`, lifespan
context managers.

## ●●● Async Python & the sync/async boundary

The trickiest conceptual thing in the repo. The RAG **core is sync**; the **edge is async**
(`docs/architecture.md §1.2`).

- A sync `RagQueryService.answer_query` is called from an async route; Phase D stops it blocking
  the loop by offloading via `await asyncio.to_thread(...)` in the query route (the core stays sync).
- The Celery worker crosses back with `asyncio.run()` (`workers/celery_app.py`). Understand why
  a sync worker calls an async repository through that bridge.
- **Import-time trap:** importing `db.session` or `workers.celery_app` builds a DB engine at
  module import. Don't import them in tests that shouldn't touch a database.

**Learn if unfamiliar:** `async`/`await`, event loop blocking, `asyncio.run`, why mixing sync
CPU/IO into an event loop serializes it.

- **The nastiest instance of this in the repo:** `detect_target_language` calls `asyncio.run`
  *inside itself* (`transliteration/detect.py:129-137`, the googletrans path). It works only
  because every caller happens to be inside `asyncio.to_thread` already. Call it from a coroutine
  and it raises `RuntimeError: asyncio.run() cannot be called from a running event loop` — and
  since the default `word-list` detector never reaches that code, the whole test suite stays green
  while production breaks. Anything touching routing must offload.

## ●●● LangGraph & the agent graph (`agent/`)

**The only orchestration** — every question, from any of the three routes, runs through it
(`docs/architecture.md §1.9`). Nothing outside `agent/` imports LangGraph.

- **State is a `TypedDict`, and LangGraph silently ignores unknown keys** a node returns. A typo'd
  key never raises; the value just stays stale. That is why nodes return `AgentUpdate`
  (a `total=False` TypedDict) rather than `dict[str, Any]` — so mypy catches it.
- **No reducers / no `Annotated` accumulators.** The graph is strictly sequential, so last-write
  wins is correct. Reaching for `operator.add` here would be cargo cult.
- **One `custom` stream channel** carries tokens *and* steps, because generation is the raw
  `openai` SDK — `stream_mode="messages"` only sees LangChain `BaseChatModel` output.
  `emit()` no-ops under `ainvoke` and off-graph, which is what lets one `generate` node serve both
  the streaming and blocking routes with no `if streaming:`.
- **mypy needs no ignores**, but two gotchas: give `CompiledStateGraph[...]` all four type args
  (bare fails `disallow_any_generics`), and import `RunnableConfig` from `langchain_core.runnables`
  — `langgraph.types` re-exports it at runtime but omits it from `__all__`.
- **Conditional edges vs branches inside nodes.** "Skip condense on a first turn" is a conditional
  *entry edge*, so it shows up in `draw_mermaid()` and can't be quietly deleted.
- **The repair loop is near-inert by default, on purpose.** `RELEVANCE_SCORE_THRESHOLD=0.0` means
  only an empty retrieval triggers a retry. Positive floors were measured and lost — the cosine
  bands for correct and incorrect retrievals overlap on XQuAD-hi, so the retry replaced correct
  answers with worse ones. The `llm` grader is worse still at 8B (81% false alarms). Before
  touching either, re-run `scripts/eval_romanized.py` and read `agent/grading/score_threshold.py`.

## ●● Evaluation harness design — the trap this repo actually fell into

Retrieval metrics are only as honest as the queries they run on, and this repo has a worked example
of getting that wrong (`docs/architecture.md §3.1`).

- **Never generate test inputs with a component under test.** `eval_romanized.py` synthesized
  romanized queries with `indic_transliteration.sanscript`, which is also the `rule-based`
  transliteration adapter — so that adapter was scored on inverting its own character mapping.
  Result: 0.950 vs google's 0.700 on identical queries, and a *feature* (`retransliterate`) built
  on the artifact and later deleted.
- **Synthetic inputs drift from real ones in ways that change the answer.** IAST-stripped
  Devanagari gives `josa narmana`; humans type `josh norman`. English loanwords staying in English
  is exactly what bge-m3 can match cross-lingually, so the synthetic version understated the real
  pipeline by a wide margin (0.669 → 0.917).
- **Disclose residual bias in the output, not the commit message.** `romanize()` still falls back
  to the rule-based scheme for ~14% of words, and every run prints that share.
- **A one-sided acceptance bar can adopt a broken component.** `LlmRelevanceGrader` fails open, so
  a judge model that 404s or times out grades everything relevant and posts a *perfect* 0%
  false-alarm rate. `scripts/eval_grader.py` therefore requires false alarms <20% **and** catches
  ≥50%. Whenever a component has a fail-safe direction, the metric must be able to see it.
- **LLM-as-judge is not free capability.** Measured on this project's endpoint: llama-3.1-8b
  false-alarms on 81% of *correct* retrievals, the 70B on 75% — 9× the parameters for 6 points.
  Test a judge before designing around one, and probe the model id first: `models.list()`
  advertises far more than a free-tier key can actually reach.
- **The habit:** before optimizing against a number, ask what would have to be true for it to be
  wrong — and check that first. Roughly a day of this project's churn traces to skipping it.

**Learn if unfamiliar:** recall@k / MRR / nDCG, distractor corpora, why a benchmark where every
question has an answer in-corpus can't measure "did retrieval fail".

**Learn if unfamiliar:** `StateGraph`, nodes vs edges, `add_conditional_edges`, cycles and
`recursion_limit`, `astream` stream modes. Skip checkpointers and `ToolNode` — this graph
deliberately uses neither (§1.9 explains why).

## ●● SQLAlchemy 2.x (async) + Alembic

- Async ORM: `AsyncSession`, `async_sessionmaker`, `Mapped[...]` / `mapped_column`.
- **Core bulk `delete()` bypasses ORM `cascade`** — this broke `DELETE` until Phase D added FK
  `ondelete="CASCADE"` (migration `0002`). Know the difference between session-level cascade and a
  bulk DML `DELETE`, and that `ondelete` is enforced by Postgres, not the ORM.
- Migrations derive the URL from `DATABASE_URL` and rewrite async→sync driver
  (`alembic/env.py`); `alembic.ini`'s URL is ignored.

**Learn if unfamiliar:** async SQLAlchemy sessions, `relationship(cascade=...)` vs bulk DML,
`ondelete="CASCADE"`, Alembic autogenerate/upgrade.

## ●● Celery + Redis

- `celery_app.py` defines the app and the `ingest_document` task; Redis is broker + result
  backend.
- The task is a thin sync wrapper doing `asyncio.run(_run_ingestion_job(...))`.
- Tests bypass Celery entirely by injecting `app.state.enqueue_ingestion`.

**Learn if unfamiliar:** Celery task definition/enqueue (`.delay`), broker vs backend, running
a worker locally.

## ●● ChromaDB & vector retrieval

- Embedded `PersistentClient`, **cosine** space, `score = 1.0 - distance`.
- Metadata must be **flat scalars**; custom fields are `meta_`-prefixed on write, unwrapped on
  read (`chroma_store.py`).
- Filtering uses Mongo-style `where` clauses (`$and`, equality) — `user_id` (Phase A) and
  `session_id` (M18, per-chat documents) scoping are both enforced this way.
- Embedded Chroma caches index segments **per process**, so a client opened before another process
  writes serves stale results (silent misses, or "Error finding id"). `ChromaVectorStore` handles
  this with **reload-on-change**: track the persist dir's newest mtime and, when it advances, clear
  Chroma's process-wide client cache and reopen. A lesson: "it works in one process" is not
  evidence it works across the api + worker split.

**Learn if unfamiliar:** dense vector retrieval, cosine similarity, ANN vs exact search,
metadata filtering, embedding dimensions (OpenAI 1536 vs bge-m3 1024 — not interchangeable in
one collection).

## ●● Multilingual NLP & embeddings — the project's whole point

- **Tokenization is script-dependent.** Whitespace splitting (`\S+`) is wrong for Chinese,
  Japanese, Thai (no inter-word spaces). Chunk size must be counted in the embedding model's
  **own tokens**, respecting its max sequence length. This is the core Phase C fix; M0
  quantified the damage (`docs/m0/report.md`).
- **Embedding models differ in contract:** `multilingual-e5-large` requires `query:` /
  `passage:` prefixes and caps at 512 tokens; `bge-m3` requires **no** prefix and handles 8192.
  Getting this wrong fails silently. bge-m3 is the selected model.
- **Cross-lingual retrieval:** a query in language X retrieving documents in language Y. The
  retention ratio (cross ÷ monolingual recall) is how we measure it.
- **Romanized script is a wall, not a dialect.** bge-m3 retrieves romanized Hindi at ~0.20 (the
  language signal lives in the script); transliterating Latin→Devanagari before embedding recovers
  it to ~0.67. The lesson from the eval: you can't fuse a raw-romanized search with a transliterated
  one and win (the raw search is unavoidable noise) — **detect** whether to transliterate instead
  (`transliteration/detect.py`), don't blend. See `docs/architecture.md §1.5b`.
- **Language detection** (`langdetect`) is unreliable on short text — hence the `min_text_length`
  fallback, and the `"unknown"`-leaks-into-the-prompt bug (fixed in Phase C). It's also
  script-based, so it can't spot *romanized* Hindi at all — that needs the function-word detector.

**Learn if unfamiliar:** subword tokenization (BPE/SentencePiece/XLM-R), sentence-transformers,
query/passage asymmetry, cross-lingual embedding spaces.

## ●● Retrieval evaluation methodology

- Metrics in `evaluation/metrics.py`: **recall@k, MRR, nDCG@k**, plus citation
  precision/recall and faithfulness (Phase B).
- **recall@1 vs recall@5 gap** diagnoses a *ranking* problem (reranker helps) vs a *finding*
  problem (retriever is broken).
- **Pre-register thresholds** before seeing results — the discipline that keeps a spike honest
  (see `docs/m0/report.md`).
- Corpus reproducibility: pinned dataset revisions + a fixed seed make regeneration
  byte-identical (`scripts/build_eval_corpus.py`).

**Learn if unfamiliar:** recall@k / MRR / nDCG definitions, LLM-as-judge, parallel eval corpora
(XQuAD), why distractors are mandatory.

## ● LLM generation & grounded answers

- `AnswerGenerator` port; one `OpenAICompatibleAnswerGenerator` serves any `chat.completions`
  endpoint (NVIDIA NIM by default, also OpenRouter/Groq/Ollama/OpenAI) — the provider is a URL
  (`GENERATION_BASE_URL`), not a code path. Zero-budget by default (free NIM tier).
- **Grounding + citations:** context chunks are numbered `[1] [2]` (`retrieval/context.py`) and
  the prompt asks the model to cite by bracket; `generation/citations.py` parses the markers and
  cites only those (the Phase B fix — the old adapter cited every retrieved chunk).
- Free-tier **rate limits** shape design (sample, don't judge all 1190 eval questions). The same
  `ChatClient` is reused by the `llm` transliteration adapter.

**Learn if unfamiliar:** RAG prompting, grounding/faithfulness, citation parsing, OpenAI-
compatible API surfaces.

## ● Transliteration (romanized-Indic query support)

- `Transliterator` **port** (`transliteration/base.py`) with swappable adapters, selected by
  `TRANSLITERATION_PROVIDER`. Default `google` (googletrans) uses a trick: `src="en", dest="hi"`
  makes Google *transliterate* romanized Hindi rather than *translate* it (a plain `dest="hi"`
  no-ops, since it detects the input as already-Hindi). It's a network call and an unofficial
  scraper, so the adapter falls back to the local rule-based transliterator on any failure.
- `indicxlit` is a local neural model (`psidharth567/indic-xlit-50M`, a Gemma-3 char-model) — the
  AI4Bharat IndicXlit proper needs `fairseq`, which won't build on Py3.13; this one loads with
  plain `transformers`. Revision-pinned, lazy-loaded, self-heals to rule-based on failure.
- **The design lesson worth internalizing:** the intuitive "search both forms and merge" loses —
  every fusion strategy dragged Hindi recall below pure transliteration because the raw search is
  irreducible noise. Deciding *whether* to transliterate (a linguistic detector) beats hedging.
- **Detector is swappable** (`TRANSLITERATION_DETECTOR`), and detection returns the *target
  language* so routing sends each query to the right script: a word list (default; ~98%/0-FP,
  Hindi-only); **MuRIL multinomial** (`google/muril-base-cased` frozen features + a hi/kn/te/other LR
  head, `muril.py` — the *right* Indic model, local, no network); or `google` (googletrans `detect()`,
  hi/kn/te, needs no training data but a network call per query). Two lessons: (1) MuRIL frozen
  features *do* separate romanized hi/kn/te linearly — a first run showing kn=0.000 was a threshold
  bug (a 0.5 max-proba floor drops correct 4-class predictions), not the model; always diagnose a
  suspiciously-bad metric before blaming the approach. (2) For romanized language-ID, char n-grams
  are a classic, even-lighter alternative that scored comparably here.
- **No romanized kn/te Q&A corpus exists** (IndicQA-romanized lacks them; FLORES-plus is gated;
  script-based sets are unloadable in `datasets` 5.x). `scripts/build_indic_romanized_eval.py`
  synthesizes one from native Wikipedia sentences + the `indic_transliteration` romanizer — the same
  trick the Hindi eval used. Knowing where free Indic data lives (and doesn't) is half the battle.

**Learn if unfamiliar:** transliteration vs translation, IAST/ITRANS schemes, `googletrans`
async API, why script (not language) is what dense retrieval keys on.

## ● Auth & security primitives

- Hand-rolled **PBKDF2-HMAC-SHA256** in `auth/security.py` (310k iterations, per-password salt,
  `hmac.compare_digest`) — no passlib. Competent, but know why constant-time compare matters.
- **JWT** (PyJWT) bearer tokens; `get_current_user` dependency.
- **Multi-tenancy** must be enforced at the data layer, not just the metadata table — the Phase
  A lesson (scope the vector store, not only Postgres).

**Learn if unfamiliar:** password hashing (salt, work factor, timing attacks), JWT
structure/validation, tenant isolation.

## ● Tooling & environment

- **Windows / PowerShell**; venv interpreter `.\.venv\Scripts\python.exe`.
- Gates: `python -m pytest` · `python -m ruff check .` (line length 100) · `python -m mypy src`.
- Docker Compose for the full stack; `scripts/smoke_test.py` for a live check.
- **GPU note:** local embedding models run on an RTX 3050 (4GB) — load models sequentially,
  they don't co-reside.

---

## Fastest path to productive

1. `docs/architecture.md` §1 — the shape.
2. `CLAUDE.md` — the conventions you must not violate.
3. `core/models.py` + one full vertical slice: `query.py` route → `RetrievalService` →
   `ChromaVectorStore`.
4. `docs/m0/report.md` — why bge-m3, and the multilingual pitfalls that motivate Phase C.
5. `docs/progress.md` — what's done and what's next.
