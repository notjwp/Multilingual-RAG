# Multilingual RAG

Production-ready multilingual retrieval-augmented generation pipeline using Python 3.13,
OpenAI, ChromaDB, and `langdetect`.

## Current Milestone

Milestone 3 adds the ingestion pipeline:

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
