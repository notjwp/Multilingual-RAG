# Multilingual RAG

Production-ready multilingual retrieval-augmented generation pipeline using Python 3.13,
OpenAI, ChromaDB, and `langdetect`.

## Current Milestone

Milestone 10 completes the first production-ready local pipeline:

- Python package layout under `src/multilingual_rag`
- Environment-driven settings
- Standard JSON logging setup
- Test and quality tooling configuration
- FastAPI application factory
- Health and readiness endpoints
- Standard API error response schema
- TXT, Markdown, HTML, PDF, and DOCX loaders
- `langdetect` language detection
- Deterministic overlapping text chunking
- Document and chunk metadata models
- Embedding provider protocol
- OpenAI embedding client adapter
- Batched document embeddings and single-query embeddings
- Structured errors for missing API keys and provider failures
- ChromaDB vector-store implementation
- Retrieval service with metadata filters
- OpenAI Responses-based answer generator
- `POST /v1/query` endpoint with answer, citations, and retrieved chunks
- Synchronous document upload, metadata lookup, and delete endpoints
- File-backed document metadata store
- Offline evaluation metrics and JSONL report CLI
- Dockerfile, Docker Compose, and HTTP smoke test script

## Planned Stack

- Python 3.13
- FastAPI for HTTP APIs
- OpenAI for embeddings and answer generation
- ChromaDB for the initial vector store
- `langdetect` for language detection
- pytest, ruff, and mypy for verification

## Local Setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` before running milestones that call
OpenAI.

## Verification

```powershell
python -m pytest
python -m ruff check .
python -m mypy src
```

## Run The API

```powershell
.\.venv\Scripts\python.exe -m uvicorn multilingual_rag.api.app:app --host 127.0.0.1 --port 8000
```

Health endpoints:

- `GET http://127.0.0.1:8000/healthz`
- `GET http://127.0.0.1:8000/readyz`

Document endpoints:

- `POST http://127.0.0.1:8000/v1/documents/upload`
- `GET http://127.0.0.1:8000/v1/documents/{document_id}`
- `DELETE http://127.0.0.1:8000/v1/documents/{document_id}`

Query endpoint:

- `POST http://127.0.0.1:8000/v1/query`

Example body:

```json
{
  "query": "What is this document about?",
  "preferred_language": "en",
  "top_k": 5,
  "filters": {
    "language": "en"
  }
}
```

The default query path requires documents to be embedded into ChromaDB first and requires
`OPENAI_API_KEY` for OpenAI embeddings and answer generation. `/v1/query` requires a bearer
token; each user only retrieves their own documents.

> **Re-index required after upgrading.** Two changes invalidate old vectors: (1) chunk vectors
> are now scoped by `user_id` (vectors written before this carry none and fail closed), and
> (2) the default embedding model is now **bge-m3 (1024-dim)** instead of OpenAI (1536-dim) —
> Chroma rejects a dimension change on an existing collection. Wipe `data/chroma` and re-ingest.
> Set `EMBEDDING_PROVIDER=openai` to keep the previous embeddings.

## Evaluation

Fixture mode scores a precomputed JSONL dataset:

```powershell
python -m multilingual_rag.evaluation.run data/eval/sample_qa.jsonl --k 2
```

Live mode runs the real retrieval pipeline (local bge-m3 embeddings + Chroma) over the XQuAD
corpus in `data/eval/xquad/` — free, no API calls (`sentence-transformers` is a core dependency):

```powershell
python -m multilingual_rag.evaluation.run --live --langs en zh --k 5
# --sample N caps distractors/queries per language for a fast smoke run (inflates recall)
```

Generation-side metrics (citation precision, faithfulness, answer language) arrive with the
free generation adapter in a later milestone; live mode currently measures retrieval only.

## Docker

```powershell
docker compose up --build
```

Run the smoke test against a running API:

```powershell
python scripts/smoke_test.py --base-url http://127.0.0.1:8000
```
