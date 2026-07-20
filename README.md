# Multilingual RAG

A multilingual retrieval-augmented generation pipeline (Python 3.13, FastAPI, ChromaDB) that
runs **free by default** — local `bge-m3` embeddings and a free OpenAI-compatible generation
endpoint (NVIDIA NIM), no paid API required. Ports-and-adapters throughout, `mypy --strict`.

## Features

- **Free, local-first core.** Default embeddings are `bge-m3` (local, 1024-dim, strong
  cross-lingual retrieval); generation targets any OpenAI-compatible endpoint (NVIDIA NIM by
  default) — switching providers is a URL change, not code.
- **Multilingual retrieval** with tokenizer-aware chunking (CJK/Thai don't split on whitespace)
  and answer-language resolution.
- **Romanized Indic queries (Hindi).** Type Hindi in the Latin alphabet
  (`bharat ki rajdhani kya hai`) and it's detected, transliterated to Devanagari, and matched
  against your native-script index. See [Romanized Hindi](#romanized-hindi-queries).
- **Authenticated, multi-tenant.** JWT bearer auth; every user retrieves only their own
  documents, enforced at the vector store, not just the metadata table.
- **Asynchronous ingestion.** Upload returns a `job_id`; a Celery worker ingests → embeds →
  indexes → records rows, and clients poll job status. Postgres is the source of truth.
- **Measurable.** A live evaluation harness (recall@k / MRR / nDCG, citation precision/recall,
  faithfulness) runs the real pipeline over the XQuAD corpus — free, no API calls.

## Stack

Python 3.13 · FastAPI · Pydantic v2 · SQLAlchemy 2 (async) + Alembic · Postgres · Celery + Redis
· ChromaDB · `sentence-transformers` (bge-m3) · `googletrans` + `indic-transliteration`
(romanized-Hindi) · pytest / ruff / mypy.

## Local setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env`. Everything runs free out of the box; set `GENERATION_API_KEY`
(a free NVIDIA NIM key from https://build.nvidia.com) to enable answer generation. Postgres and
Redis are needed for anything touching documents or auth — `docker compose up postgres redis` is
the quickest way.

## Verification

```powershell
python -m pytest
python -m ruff check .          # add --fix to autofix
python -m mypy src
```

Model- and Postgres-backed tests are opt-in / auto-skipped: `RUN_MODEL_TESTS=1` exercises bge-m3;
the DB-layer tests run when a local Postgres is reachable.

## Run the stack

```powershell
python -m uvicorn multilingual_rag.api.app:app --host 127.0.0.1 --port 8000
celery -A multilingual_rag.workers.celery_app.celery_app worker --loglevel=INFO
alembic upgrade head
docker compose up --build            # postgres + redis + api + worker
```

Endpoints (all under `/v1`, bearer token required except health):

- `GET /healthz` · `GET /readyz`
- `POST /v1/auth/signup` · `POST /v1/auth/login`
- `POST /v1/documents/upload` → `{ "job_id": ... }` (async) · `GET /v1/ingestion-jobs/{job_id}`
- `GET /v1/documents/{id}` · `DELETE /v1/documents/{id}`
- `POST /v1/query`

Example query:

```json
{ "query": "bharat ki rajdhani kya hai", "top_k": 5 }
```

The response carries the answer, citations, retrieved chunks, and — when the query was romanized
Hindi — `transliterated_query` and `transliteration_applied`.

> **Re-index required after upgrading.** Vectors are scoped by `user_id` (older vectors carry
> none and fail closed), and the default embedding model is **bge-m3 (1024-dim)**, not OpenAI
> (1536-dim) — Chroma rejects a dimension change on an existing collection. Wipe `data/chroma`
> and re-ingest, or set `EMBEDDING_PROVIDER=openai` to keep OpenAI embeddings.

## Romanized Hindi queries

`bge-m3` can't retrieve from romanized Hindi — the language signal lives in the script, so a
Latin-typed query collapses to ~0.20 recall against a native-Devanagari index. The query path
detects romanized Hindi (distinctly-Hindi function words) and transliterates it to Devanagari
before embedding, recovering recall to ~0.67 (3.3×) while leaving plain English untouched.

Configured via `TRANSLITERATION_PROVIDER` (`.env`):

- `google` (default) — googletrans; best quality, free, a network call per query, with a local
  rule-based fallback baked in.
- `indicxlit` — a local offline neural model (no network).
- `rule-based` — `indic-transliteration`, instant and offline, lower quality.
- `llm` — reuse the generation endpoint (costs credits). `off` — disable.

The *detector* is swappable via `TRANSLITERATION_DETECTOR`:

- `word-list` (default) — a fast local function-word check, ~98% recall / 0 false-positives. Hindi only.
- `muril` — a frozen `google/muril-base-cased` feature extractor + a small LogisticRegression head
  (`scripts/train_romanized_detector.py`); lazy on CPU, word-list fallback. Hindi only.
- `google` — googletrans `detect()` identifies the language, enabling **Kannada/Telugu**. Set
  `TRANSLITERATION_LANGUAGES=hi,kn,te`. Validated (Wikipedia-derived eval,
  `scripts/build_indic_romanized_eval.py`): kn/te romanized→native recovery ~0.59→~0.97, 0 English
  false-positives. Opt-in — costs a network call per query.

Design details in `docs/architecture.md §1.5b`; the motivating spike in `docs/indic-romanized-spike.md`.

## Evaluation

```powershell
# Fixture mode: score a precomputed JSONL dataset.
python -m multilingual_rag.evaluation.run data/eval/sample_qa.jsonl --k 2

# Live mode: the real pipeline (bge-m3 + Chroma) over the XQuAD corpus — free, no API calls.
python -m multilingual_rag.evaluation.run --live --langs en zh --k 5
# --sample N caps distractors/queries per language for a fast smoke run (inflates recall).

# Romanized-Hindi eval: native / romanized-raw / transliterated / shipped conditions.
python scripts/eval_romanized.py --sample 150

# Regenerate the eval corpus (pinned dataset revisions, byte-identical).
python scripts/build_eval_corpus.py
```

## Docker & smoke test

```powershell
docker compose up --build
python scripts/smoke_test.py --base-url http://127.0.0.1:8000
```

## Documentation

- `docs/architecture.md` — HLD/LLD, request flows, known defects.
- `docs/progress.md` — what's done and what's next.
- `docs/skills.md` — the technical domains and repo-specific gotchas.
- `docs/m0/report.md` — the cross-lingual embedding spike (why bge-m3).
- `docs/indic-romanized-spike.md` — the romanized-Hindi spike.
