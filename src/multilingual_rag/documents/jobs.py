"""Document ingestion job execution."""

from __future__ import annotations

import contextlib
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import DocumentRecord
from multilingual_rag.db.models import IngestionJob
from multilingual_rag.documents.repository import DocumentRepository, IngestionJobRepository
from multilingual_rag.embeddings.base import EmbeddingProvider
from multilingual_rag.embeddings.factory import build_embedding_provider
from multilingual_rag.ingestion.service import IngestionService
from multilingual_rag.vectorstores.base import VectorStore
from multilingual_rag.vectorstores.factory import build_vector_store


def _chunk_metadata(chunks: object) -> list[dict[str, object]]:
    return [
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
        for chunk in chunks  # type: ignore[attr-defined]
    ]


async def run_ingestion_job(
    *,
    settings: Settings,
    session: AsyncSession,
    job_id: str,
    ingestion_service: IngestionService | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
) -> str:
    """Run a queued ingestion job and return the indexed document ID.

    Dependencies default to the real components but are injectable for testing. The write order
    (DB rows, then vectors, then one commit; compensating vector delete on failure) keeps
    Postgres and Chroma from drifting into an orphan-vectors or document-without-vectors state.
    """
    ingestion_service = ingestion_service or IngestionService(settings)
    embedding_provider = embedding_provider or build_embedding_provider(settings)
    vector_store = vector_store or build_vector_store(settings)

    job_repository = IngestionJobRepository(session)
    job = await session.get(IngestionJob, job_id)
    if job is None:
        raise ValueError(f"Ingestion job not found: {job_id}")

    # Capture into locals before any rollback — a rolled-back ORM object can't be read
    # (async lazy-load fails), and the failure-cleanup path needs these.
    user_id = job.user_id
    session_id = job.session_id  # the chat this document is scoped to (M18), or None
    file_path = Path(job.file_path)

    await job_repository.mark_running(job_id)
    await session.commit()

    # Set once the document id is known, so failure cleanup can remove any vectors written.
    document_id: str | None = None
    try:
        ingestion_result = ingestion_service.ingest_file(
            file_path, user_id=user_id, session_id=session_id
        )
        document_id = ingestion_result.document.document_id
        embeddings = embedding_provider.embed_documents(
            tuple(chunk.text for chunk in ingestion_result.chunks)
        )

        record = DocumentRecord(
            document=ingestion_result.document,
            chunk_count=len(ingestion_result.chunks),
        )
        await DocumentRepository(session).save(
            record,
            user_id=user_id,
            session_id=session_id,
            file_path=file_path,
            filename=file_path.name,
            file_size_bytes=file_path.stat().st_size,
            chunk_metadata=_chunk_metadata(ingestion_result.chunks),
        )
        vector_store.upsert_chunks(
            ingestion_result.chunks, embeddings, user_id=user_id, session_id=session_id
        )
        await job_repository.mark_succeeded(job_id=job_id, document_id=document_id)
        await session.commit()
        return document_id
    except Exception as exc:
        await session.rollback()
        if document_id is not None:
            # Best-effort: drop any vectors that landed, so a failed job leaves nothing behind.
            with contextlib.suppress(Exception):
                vector_store.delete_document(document_id, user_id=user_id, session_id=session_id)
        await job_repository.mark_failed(job_id=job_id, error_message=str(exc))
        await session.commit()
        raise
