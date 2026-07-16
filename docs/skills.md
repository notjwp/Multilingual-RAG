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

- A sync `RagQueryService.answer_query` is called from an async route — fine today, but it
  blocks the event loop (a known defect; Phase D fixes it).
- The Celery worker crosses back with `asyncio.run()` (`workers/celery_app.py`). Understand why
  a sync worker calls an async repository through that bridge.
- **Import-time trap:** importing `db.session` or `workers.celery_app` builds a DB engine at
  module import. Don't import them in tests that shouldn't touch a database.

**Learn if unfamiliar:** `async`/`await`, event loop blocking, `asyncio.run`, why mixing sync
CPU/IO into an event loop serializes it.

## ●● SQLAlchemy 2.x (async) + Alembic

- Async ORM: `AsyncSession`, `async_sessionmaker`, `Mapped[...]` / `mapped_column`.
- **Core bulk `delete()` bypasses ORM `cascade`** — this is an active bug in
  `documents/repository.py` (Phase D). Know the difference between session-level cascade and a
  bulk DML `DELETE`.
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
- Filtering uses Mongo-style `where` clauses (`$and`, equality) — Phase A adds server-side
  `user_id` scoping this way.
- Embedded Chroma is **not safe for concurrent multi-process writers** (Phase D → server mode).

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
- **Language detection** (`langdetect`) is unreliable on short text — hence the `min_text_length`
  fallback, and the `"unknown"`-leaks-into-the-prompt bug (Phase C).

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

- `AnswerGenerator` port; OpenAI Responses API adapter today, moving to a free-tier
  OpenAI-compatible endpoint (OpenRouter/Groq) under the zero-budget constraint (Phase C).
- **Grounding + citations:** context chunks are numbered `[1] [2]` (`retrieval/context.py`) and
  the prompt asks the model to cite by bracket — but the current adapter ignores the markers
  and cites everything (Phase B fix).
- Free-tier **rate limits** shape design (sample, don't judge all 1190 eval questions).

**Learn if unfamiliar:** RAG prompting, grounding/faithfulness, citation parsing, OpenAI-
compatible API surfaces.

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
