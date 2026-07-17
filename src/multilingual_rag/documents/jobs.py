"""Document ingestion job execution."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import DocumentRecord
from multilingual_rag.db.models import IngestionJob
from multilingual_rag.documents.repository import DocumentRepository, IngestionJobRepository
from multilingual_rag.embeddings.factory import build_embedding_provider
from multilingual_rag.ingestion.service import IngestionService
from multilingual_rag.vectorstores.chroma_store import ChromaVectorStore


async def run_ingestion_job(*, settings: Settings, session: AsyncSession, job_id: str) -> str:
    """Run a queued ingestion job and return the indexed document ID."""
    job_repository = IngestionJobRepository(session)
    job = await session.get(IngestionJob, job_id)
    if job is None:
        raise ValueError(f"Ingestion job not found: {job_id}")

    await job_repository.mark_running(job_id)
    await session.commit()

    try:
        ingestion_result = IngestionService(settings).ingest_file(Path(job.file_path))
        embeddings = build_embedding_provider(settings).embed_documents(
            tuple(chunk.text for chunk in ingestion_result.chunks)
        )
        ChromaVectorStore(settings).upsert_chunks(
            ingestion_result.chunks, embeddings, user_id=job.user_id
        )

        record = DocumentRecord(
            document=ingestion_result.document,
            chunk_count=len(ingestion_result.chunks),
        )
        await DocumentRepository(session).save(
            record,
            user_id=job.user_id,
            file_path=Path(job.file_path),
            filename=Path(job.file_path).name,
            file_size_bytes=Path(job.file_path).stat().st_size,
            chunk_metadata=[
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "language": chunk.language,
                    "source": chunk.source,
                    "page": chunk.page,
                    "token_count": chunk.token_count,
                    "checksum": chunk.checksum,
                    "metadata": chunk.metadata,
                }
                for chunk in ingestion_result.chunks
            ],
        )
        await job_repository.mark_succeeded(job_id=job_id, document_id=record.document.document_id)
        await session.commit()
        return record.document.document_id
    except Exception as exc:
        await session.rollback()
        await job_repository.mark_failed(job_id=job_id, error_message=str(exc))
        await session.commit()
        raise
