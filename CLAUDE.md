# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Setup (Windows / PowerShell — the venv interpreter is `.\.venv\Scripts\python.exe`):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Verification — all three must pass:

```powershell
python -m pytest
python -m ruff check .          # add --fix to autofix imports/formatting
python -m mypy src
```

Single test / subset:

```powershell
python -m pytest tests/unit/test_chunker.py
python -m pytest tests/unit/test_chunker.py::test_name
python -m pytest -k "retrieval"
```

Run the stack:

```powershell
python -m uvicorn multilingual_rag.api.app:app --host 127.0.0.1 --port 8000
celery -A multilingual_rag.workers.celery_app.celery_app worker --loglevel=INFO
alembic upgrade head
python -m multilingual_rag.evaluation.run data/eval/sample_qa.jsonl --k 2
docker compose up --build            # postgres + redis + api + worker
python scripts/smoke_test.py --base-url http://127.0.0.1:8000
```

Postgres and Redis must be running for anything touching documents or auth; `docker compose up postgres redis` is the quickest way. `OPENAI_API_KEY` is required for embedding/generation paths.

## Architecture

Request → route → service → protocol-typed adapter. Layers only depend downward, and every external system sits behind a `Protocol` port.

**Ports and adapters.** `embeddings/base.py`, `vectorstores/base.py`, `generation/base.py`, `transliteration/base.py`, `retrieval/base.py`, and `agent/grading/base.py` each define a `Protocol` (`EmbeddingProvider`, `VectorStore`, `AnswerGenerator` + `StreamClient`, `Transliterator`, `Retriever`, `RelevanceGrader`); `bge_embeddings.py`, `chroma_store.py`, `openai_compatible_generator.py`, the `transliteration/` adapters, `RetrievalService`, and the two graders are the concretes. Each has a `build_*` factory that selects the adapter from `Settings`. Services receive these through keyword-only constructor injection and never import an adapter directly — depend on the protocol so tests can pass fakes.

**Sync core, async edge.** The RAG core — ingestion, chunking, embedding, Chroma, retrieval — is synchronous; the API, DB session, and repository layer are `async`. Keep new core logic sync; keep `async` at the HTTP/DB boundary. **The orchestrator is the exception:** the agent graph's nodes are async and push the sync core down with `asyncio.to_thread` *inside the node that blocks*, rather than a route offloading a whole sync service. Two nodes need it — `retrieve` (bge-m3 + Chroma) and `route_language`, because `detect_target_language` calls `asyncio.run` internally on the `google` path. The Celery task still bridges the other way with `asyncio.run()`.

**The agent graph (`agent/`) is the only orchestration.** All three entry points — `POST /v1/query`, blocking chat, streaming chat — go through `agent/graph.py::RagGraph` (`answer()` / `stream()`), reached via `api/dependencies.py::get_rag_graph`. Nothing outside `agent/` imports LangGraph. Shape: `condense → route_language → retrieve → grade`, then `generate → ground_check`, or `repair → retrieve` (a real cycle), or `generate_no_context` when repairs run out. `ground_check` runs on every answered turn but returns immediately unless `GROUNDING_GATE` is set (it isn't by default) — and note the longest gated path uses the recursion limit *exactly*, so adding a node breaks `test_the_longest_gated_path_stays_inside_the_recursion_limit` rather than silently truncating a rare branch. `repair` (`agent/repair.py::choose_repair`, a pure function) is the differentiated part: on a weak retrieval it retries the raw untransliterated query, re-routes to another Indic language, or LLM-rewrites — in that order. Deliberately **no tools and no checkpointer**: retrieval is a node, so the model can never author `user_id`/`session_id` (tenancy is structural, not prompt-enforced), and history already lives in `messages`. Events (`agent/events.py`) go out over LangGraph's single `custom` stream channel via `emit()`, which no-ops under `ainvoke` and off-graph — that is why `generate` always streams internally and blocking callers just collect, with no `if streaming:` anywhere.

**The default grader is `llm`, and it was chosen on refusal quality, not retrieval.** It refuses ~70% of *answerable* questions and costs ~3 provider calls per turn — both are known and accepted, because the alternative fabricates a cited answer 61% of the time on out-of-corpus questions (see the refusal section below). This is a recent flip; `score-threshold` was the default through M19 and most recorded retrieval numbers were measured under it. **Anything you read in this repo comparing agent conditions on XQuAD is a `score-threshold` number** — that benchmark has no unanswerable questions, so the current default scores worse there by construction and the comparison is not a regression.

**The repair loop is measured at parity, and `score-threshold`'s floor is deliberately inert.** `RELEVANCE_SCORE_THRESHOLD=0.0` means only an *empty* retrieval is graded weak, so under that grader the loop almost never fires. Don't raise it without re-running `scripts/eval_romanized.py`: every positive floor tried lost to having no agent at all (best 0.767 vs 0.800) because bge-m3's cosine bands for correct and incorrect retrievals overlap. Generation answers from `best_context` — the best-graded attempt, not the last — so a repair can never make things worse; it is adopted only if it flips weak→relevant, never on a higher score. `agentic` is the fifth condition in `eval_romanized.py`, which drives the real graph with generation stubbed — note it reads `settings.relevance_grader`, so it now makes real provider calls unless you pass `--grader score-threshold`.

**The eval synthesizes its romanized queries — check `evaluation/romanization.py` before trusting any Indic number.** It must never generate queries with the same library as an adapter under test; it did, and that scored `rule-based` 0.950 vs google 0.700 on identical input. Queries now come from human-written romanizations (`data/eval/romanization_hi.json`, built by `scripts/build_romanization_lexicon.py`), with the sanscript-fallback share printed every run. Consequence: numbers predating that fix are superseded — shipped romanized retention is **0.852** (full corpus, 150 queries / 20,240 docs, `data/eval/reports/hi-full-baseline.json`), not the recorded 0.669. The 0.747 bar is replaced by **0.80**, and the eval gates **only full-corpus runs** because sampling inflates retention (3,240 docs scored 0.917 where 20,240 scored 0.852). kn/te lexicons exist but cover only ~43% of their vocabulary, so those numbers stay directional.

**Two things that will waste your time if you don't know them.** (1) **No reachable LLM judge is *accurate*** — llama-3.1-8b false-alarms on 81% of correct retrievals, the 70B on 75%, the 3.3-70B times out, and most catalog ids 404 on a free-tier key. Run `scripts/eval_grader.py --model X` before believing any of them, and note the bar is two-sided: the grader fails open, so a model that 404s posts a perfect 0% false-alarm rate. This is *not* a contradiction of `RELEVANCE_GRADER=llm` being the default — that 81% false-alarm rate is precisely the 70% refusal rate, shipped knowingly because the alternative failure (a fabricated citation) was judged worse. A better judge model improves the default directly; it is the single highest-leverage upgrade available here. (2) **`langdetect` calls romanized Hindi "Swahili"** — never trust `RetrievalContext.query_language` for a Latin-script query; `route.target_language` is the authority (`RagNodes._response_language`).

**The hallucination/refusal tradeoff has no free point, and three attempts to find one have failed.** Measured by `scripts/eval_refusal.py`, the only harness here that asks *unanswerable* questions (XQuAD can't: every question has an answer, so recall@5 0.852 and a majority hallucination rate coexist happily). 20 questions per set, llama-3.1-8b:

| config | fabricates | refuses answerable | calls/turn |
|---|---|---|---|
| `RELEVANCE_GRADER=llm` (**default**) | 0% | 70% | ~3 |
| `RELEVANCE_GRADER=score-threshold` | 61% | 21% | 1 |
| `GROUNDING_GATE=true` | 40% | 55% | 2 |

Three dead ends, all measured and all reverted or shelved — **don't re-propose any of them** without reading `docs/architecture.md §1.9a`. (1) Keeping the llm grader's judgement but answering from `best_context` anyway: hallucination returns immediately, 0% → 55%. It prevents fabrication *by refusing*, and that is the whole mechanism. (2) `GROUNDING_GATE`, a post-generation faithfulness check on the *answer* rather than the retrieval: **dominated at every query mix** — better than `score-threshold` only below a 38% answerable share, better than `llm` only above 73%, ranges that don't overlap. Off by default, kept only as tested infrastructure a stronger judge could revive. (3) Reframing the grader prompt from a set-level YES/NO to a passage *selection* ("which help? numbers or NONE"), the obvious inference from the judge scoring 7/8 on single passages and failing on mixed sets: false alarms 81% → 56% and refusals 70% → 50%, but fabrication **0% → 20%**. It made the judge more *permissive*, not more *accurate* — it shed right weak grades along with wrong ones. Reverted; the prompt is pinned by `test_the_grader_asks_for_a_set_level_verdict_not_a_passage_selection`.

**The general lesson from (3), which cost a full measurement cycle:** "the policy is unchanged, so hallucination cannot rise" is *false reasoning*. Policy decides what happens given a weak grade; it does not fix how often weak grades occur. Any change to the judge moves both rates, so anything touching `grading/` needs `eval_refusal.py`, not just `eval_grader.py`.

The two graders cross at roughly a **55% answerable share**, so the right default is a property of the corpus, not of the code. n=20 on all of these; treat single-digit differences as noise.

**Agent steps are ephemeral.** `Step` events stream to the client as SSE `event: step` frames (running/done pairs sharing an `id`, so the UI upserts) and are **never persisted**. A reloaded chat shows the answer and its citations only. Don't add a `message_steps` table without a reason.

**`RetrievalService.route()`** is split out of `retrieve` so the graph can make the transliteration decision as its own step and re-make it differently on a retry. `retrieve(..., route=None)` decides for itself, reproducing the old behaviour exactly — which is why the existing retrieval tests and both eval harnesses were untouched by the split.

**One document path.** Documents go through `DatabaseDocumentIndexingService` (Postgres repositories + Celery). The legacy `DocumentIndexingService` + `DocumentStore` (a JSON file, no user scoping) was removed in Phase D — there is now a single source of truth.

**Per-chat documents (M18).** Documents are scoped to a **single chat**, not the whole user: a file uploaded into a chat only grounds *that* chat's answers. `documents`/`ingestion_jobs` carry a nullable `session_id` FK (`ondelete=CASCADE`, so deleting a chat drops its docs), the dedup constraint is `(user_id, session_id, checksum)`, and the content-addressed `document_id` folds in `session_id`. A `session_id` threads through the vector store (`VectorStore` methods take `session_id`; the Chroma adapter AND-s a `session_id` metadata filter into the `where` clause and folds it into the storage id) and through retrieval → the agent graph's `answer()` / `stream()` (both take `session_id`, carried in `AgentState`) so a chat retrieves only its own chunks. There is no user-wide document library and no global `/v1/documents` route.

**Upload is asynchronous.** `POST /v1/chats/{chat_id}/documents` (in `api/routes/chat_documents.py`) verifies chat ownership, saves bytes to `raw_document_directory`, creates a `queued` ingestion job row scoped to the chat, enqueues Celery, and returns a `job_id` — it does *not* index inline. The worker runs `documents/jobs.py::run_ingestion_job`: ingest → embed → vector upsert (scoped by `user_id`+`session_id`) → write `documents`/`document_chunks` rows → mark succeeded/failed. Clients poll `GET /v1/ingestion-jobs/{job_id}`. The `document_chunks` table mirrors vector metadata for traceability, so chunk writes must stay in sync with vector upserts.

**Romanized-Indic queries.** `RetrievalService.retrieve` detects romanized Hindi (`transliteration/detect.py::is_romanized_indic`) and, when detected, transliterates the query to Devanagari before embedding, so it matches the native-script index. Plain English is left on the raw path. Detection, not "search both and merge" — the eval proved fusion loses (the raw romanized search is irreducible noise). Provider selected by `TRANSLITERATION_PROVIDER` (default `google`/googletrans with a local rule-based fallback). `TRANSLITERATION_DETECTOR` picks the detector, and detection returns the *target language* (`detect_target_language -> str|None`) so `RetrievalService` transliterates to the right script: `word-list` (default — a distinctly-Hindi function-word check, ~98% recall/0 FP, fast/local, **Hindi only**), `muril` (opt-in — a frozen `google/muril-base-cased` feature extractor + a committed **multinomial** LR head classifying **hi/kn/te/other**, `transliteration/muril.py` + `romanized_indic_detector.joblib`; **local**, lazy CPU, word-list fallback), or `google` (opt-in — googletrans `detect()`, also hi/kn/te, a network call per query). Enable kn/te with `TRANSLITERATION_LANGUAGES=hi,kn,te` + `muril` or `google`. kn/te are validated (Wikipedia-derived eval, `scripts/build_indic_romanized_eval.py`: kn 0.96 / te ~0.97, 0 English FP) but opt-in — the default stays Hindi/word-list, no model, no network.

**Identity.** JWT bearer via `auth/dependencies.py::get_current_user`. Password hashing is hand-rolled PBKDF2-HMAC-SHA256 in `auth/security.py` (`pbkdf2_sha256$iterations$salt$digest`) — no passlib.

## Conventions

**`app.state` is the injection seam.** This codebase does not use FastAPI `dependency_overrides`. Routes call a module-level `get_*` helper (`get_rag_graph(request)` in `api/dependencies.py`, `get_document_service(request, session)`) that returns `request.app.state.<attr>` when set and otherwise constructs the real dependency. Tests attach fakes to `app.state`:

```python
app = create_app(Settings(environment="test"))
app.state.rag_graph = FakeRagGraph()                 # async answer() -> AgentResult; stream()
app.state.document_service = FakeDocumentService()
app.state.current_user = UserRecord(user_id="user-1", email="user@example.com")
app.state.enqueue_ingestion = enqueued_jobs.append   # bypasses Celery
```

Recognized attrs: `settings`, `rag_graph`, `chat_service`, `document_service`, `current_user`, `enqueue_ingestion`. (`query_service` and `streaming_answerer` are gone — one graph replaced both.) When adding a route with a new dependency, follow this pattern — declare a `Protocol` for the service, add the `get_*` fallback helper, and hang the override off `app.state`. There is no `conftest.py`; tests build their own app and fakes are plain classes, not mocks.

**Errors.** Raise `AppError(message, code="snake_case_code", status_code=...)`, never `HTTPException`. A single handler in `api/app.py` renders it as `ErrorResponse`. The `code` is part of the API contract.

**Domain models.** Everything in `core/models.py` is a frozen pydantic model (`ConfigDict(frozen=True)`) and collections are `tuple[...]`, not `list[...]` — this propagates through service signatures and response models.

**Settings.** Injected as a `Settings` object, not read from env at use sites. Routes reach it via `cast(Settings, request.app.state.settings)`; services take it as a constructor arg. `get_settings()` is `lru_cache`d — construct `Settings(...)` explicitly in tests rather than mutating env.

**Import-time side effects.** `db/session.py` and `workers/celery_app.py` call `get_settings()` and create engines at module import. Importing anything that transitively pulls in `db.session` reads `.env` and constructs an async engine, so avoid importing them in tests that shouldn't need a database.

**mypy is `strict = true`.** Untyped third-party libs (celery, chromadb) need `# type: ignore[...]` with the specific code. Keep `ruff` line length at 100. **LangGraph needs none** — it ships `py.typed`, and `add_node` with a partial-`TypedDict` return plus `add_conditional_edges` with a `Literal`-annotated router both type-check clean. Two things to know: supply all four type args to `CompiledStateGraph[...]` (bare fails `disallow_any_generics`), and import `RunnableConfig` from `langchain_core.runnables` — `langgraph.types` re-exports it at runtime but omits it from `__all__`.

**Package `__init__.py` files are docstring-only.** No re-exports anywhere — import from the submodule. This is not cosmetic: re-exports in `agent/__init__.py` created an import cycle (`generation.streaming` → `agent.events` → `agent/__init__` → `agent.factory` → … → `generation.streaming`).

Migrations: `alembic/env.py` ignores the `sqlalchemy.url` in `alembic.ini` and derives the URL from `get_settings().database_url`, rewriting the async driver to sync (`postgresql+asyncpg` → `postgresql`). Configure migrations through `DATABASE_URL`.

Chroma specifics (`vectorstores/chroma_store.py`): cosine space, `score = 1.0 - distance`, and metadata must be flat scalars — custom chunk metadata is stored prefixed with `meta_` and unwrapped on read. **Multi-process safety:** embedded Chroma caches index segments per process, so the API's client would go stale after the Celery worker writes (silent wrong results / "Error finding id"). The adapter guards this with *reload-on-change* — it tracks the persist dir's newest mtime and, when it advances, clears Chroma's process-wide client cache and reopens (all ops serialized under a lock). Covered by `tests/integration/test_chroma_multiprocess.py`; no Chroma server required.

The `chat_sessions`, `messages`, and `message_citations` tables in `db/models.py` are unused placeholders for a future milestone, not dead code.
