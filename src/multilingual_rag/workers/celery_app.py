"""Celery application and tasks."""

from __future__ import annotations

import asyncio

from celery import Celery  # type: ignore[import-untyped]
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from multilingual_rag.core.config import get_settings
from multilingual_rag.documents.jobs import run_ingestion_job

settings = get_settings()

celery_app = Celery(
    "multilingual_rag",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)


@celery_app.task(name="multilingual_rag.ingest_document")  # type: ignore[untyped-decorator]
def ingest_document(job_id: str) -> str:
    """Run document ingestion in a worker process."""
    return asyncio.run(_run_ingestion_job(job_id))


async def _run_ingestion_job(job_id: str) -> str:
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        return await run_ingestion_job(settings=settings, session=session, job_id=job_id)
