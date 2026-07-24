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

**Ports and adapters.** `embeddings/base.py`, `vectorstores/base.py`, `generation/base.py`, and `transliteration/base.py` each define a `Protocol` (`EmbeddingProvider`, `VectorStore`, `AnswerGenerator`, `Transliterator`); `bge_embeddings.py`, `chroma_store.py`, `openai_compatible_generator.py`, and the `transliteration/` adapters are the concretes. Each has a `build_*` factory that selects the adapter from `Settings`. Services receive these through keyword-only constructor injection and never import an adapter directly — depend on the protocol so tests can pass fakes.

**Sync core, async edge.** The whole RAG core — ingestion, chunking, embedding, Chroma, retrieval, generation — is synchronous. Only the API, DB session, and repository layer is `async`. So `RagQueryService.answer_query` is a sync method called from an async route, and the Celery task bridges back with `asyncio.run()` in `workers/celery_app.py`. Keep new core logic sync; keep `async` at the HTTP/DB boundary.

**One document path.** Documents go through `DatabaseDocumentIndexingService` (Postgres repositories + Celery). The legacy `DocumentIndexingService` + `DocumentStore` (a JSON file, no user scoping) was removed in Phase D — there is now a single source of truth.

**Per-chat documents (M18).** Documents are scoped to a **single chat**, not the whole user: a file uploaded into a chat only grounds *that* chat's answers. `documents`/`ingestion_jobs` carry a nullable `session_id` FK (`ondelete=CASCADE`, so deleting a chat drops its docs), the dedup constraint is `(user_id, session_id, checksum)`, and the content-addressed `document_id` folds in `session_id`. A `session_id` threads through the vector store (`VectorStore` methods take `session_id`; the Chroma adapter AND-s a `session_id` metadata filter into the `where` clause and folds it into the storage id) and through retrieval → `RagQueryService.answer` / `StreamingAnswerGenerator.stream` (both take `session_id`) so a chat retrieves only its own chunks. There is no user-wide document library and no global `/v1/documents` route.

**Upload is asynchronous.** `POST /v1/chats/{chat_id}/documents` (in `api/routes/chat_documents.py`) verifies chat ownership, saves bytes to `raw_document_directory`, creates a `queued` ingestion job row scoped to the chat, enqueues Celery, and returns a `job_id` — it does *not* index inline. The worker runs `documents/jobs.py::run_ingestion_job`: ingest → embed → vector upsert (scoped by `user_id`+`session_id`) → write `documents`/`document_chunks` rows → mark succeeded/failed. Clients poll `GET /v1/ingestion-jobs/{job_id}`. The `document_chunks` table mirrors vector metadata for traceability, so chunk writes must stay in sync with vector upserts.

**Romanized-Indic queries.** `RetrievalService.retrieve` detects romanized Hindi (`transliteration/detect.py::is_romanized_indic`) and, when detected, transliterates the query to Devanagari before embedding, so it matches the native-script index. Plain English is left on the raw path. Detection, not "search both and merge" — the eval proved fusion loses (the raw romanized search is irreducible noise). Provider selected by `TRANSLITERATION_PROVIDER` (default `google`/googletrans with a local rule-based fallback). `TRANSLITERATION_DETECTOR` picks the detector, and detection returns the *target language* (`detect_target_language -> str|None`) so `RetrievalService` transliterates to the right script: `word-list` (default — a distinctly-Hindi function-word check, ~98% recall/0 FP, fast/local, **Hindi only**), `muril` (opt-in — a frozen `google/muril-base-cased` feature extractor + a committed **multinomial** LR head classifying **hi/kn/te/other**, `transliteration/muril.py` + `romanized_indic_detector.joblib`; **local**, lazy CPU, word-list fallback), or `google` (opt-in — googletrans `detect()`, also hi/kn/te, a network call per query). Enable kn/te with `TRANSLITERATION_LANGUAGES=hi,kn,te` + `muril` or `google`. kn/te are validated (Wikipedia-derived eval, `scripts/build_indic_romanized_eval.py`: kn 0.96 / te ~0.97, 0 English FP) but opt-in — the default stays Hindi/word-list, no model, no network.

**Identity.** JWT bearer via `auth/dependencies.py::get_current_user`. Password hashing is hand-rolled PBKDF2-HMAC-SHA256 in `auth/security.py` (`pbkdf2_sha256$iterations$salt$digest`) — no passlib.

## Conventions

**`app.state` is the injection seam.** This codebase does not use FastAPI `dependency_overrides`. Routes call a module-level `get_*_service(request, ...)` helper (`get_query_service(request)`, `get_document_service(request, session)`) that returns `request.app.state.<attr>` when set and otherwise constructs the real dependency. Tests attach fakes to `app.state`:

```python
app = create_app(Settings(environment="test"))
app.state.query_service = FakeQueryService()
app.state.document_service = FakeDocumentService()
app.state.current_user = UserRecord(user_id="user-1", email="user@example.com")
app.state.enqueue_ingestion = enqueued_jobs.append   # bypasses Celery
```

Recognized attrs: `settings`, `query_service`, `document_service`, `current_user`, `enqueue_ingestion`. When adding a route with a new dependency, follow this pattern — declare a `Protocol` for the service, add the `get_*` fallback helper, and hang the override off `app.state`. There is no `conftest.py`; tests build their own app and fakes are plain classes, not mocks.

**Errors.** Raise `AppError(message, code="snake_case_code", status_code=...)`, never `HTTPException`. A single handler in `api/app.py` renders it as `ErrorResponse`. The `code` is part of the API contract.

**Domain models.** Everything in `core/models.py` is a frozen pydantic model (`ConfigDict(frozen=True)`) and collections are `tuple[...]`, not `list[...]` — this propagates through service signatures and response models.

**Settings.** Injected as a `Settings` object, not read from env at use sites. Routes reach it via `cast(Settings, request.app.state.settings)`; services take it as a constructor arg. `get_settings()` is `lru_cache`d — construct `Settings(...)` explicitly in tests rather than mutating env.

**Import-time side effects.** `db/session.py` and `workers/celery_app.py` call `get_settings()` and create engines at module import. Importing anything that transitively pulls in `db.session` reads `.env` and constructs an async engine, so avoid importing them in tests that shouldn't need a database.

**mypy is `strict = true`.** Untyped third-party libs (celery, chromadb) need `# type: ignore[...]` with the specific code. Keep `ruff` line length at 100.

Migrations: `alembic/env.py` ignores the `sqlalchemy.url` in `alembic.ini` and derives the URL from `get_settings().database_url`, rewriting the async driver to sync (`postgresql+asyncpg` → `postgresql`). Configure migrations through `DATABASE_URL`.

Chroma specifics (`vectorstores/chroma_store.py`): cosine space, `score = 1.0 - distance`, and metadata must be flat scalars — custom chunk metadata is stored prefixed with `meta_` and unwrapped on read.

The `chat_sessions`, `messages`, and `message_citations` tables in `db/models.py` are unused placeholders for a future milestone, not dead code.
