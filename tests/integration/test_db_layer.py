"""DB-layer tests against real Postgres — the coverage that was missing.

The whole persistence layer (DocumentRepository, IngestionJobRepository) had zero tests because
every route test injects a fake. That gap hid the psycopg2 miss and the broken DELETE. These run
only when Postgres is reachable (skipped otherwise, so the offline suite stays green), against a
throwaway `multilingual_rag_test` database.

Several tests start RED and pin the Phase-D fixes (D4 delete cascade, D7 content checksum).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import (
    DocumentChunk as DomainChunk,
)
from multilingual_rag.core.models import (
    DocumentMetadata,
    DocumentRecord,
    IngestionResult,
)
from multilingual_rag.db.base import Base
from multilingual_rag.db.models import Document, DocumentChunk, DocumentFile, User
from multilingual_rag.documents.jobs import run_ingestion_job
from multilingual_rag.documents.repository import DocumentRepository, IngestionJobRepository

MAINT_URL = "postgresql://postgres:postgres@localhost:5432/postgres"
TEST_DB = "multilingual_rag_test"
ASYNC_URL = f"postgresql+asyncpg://postgres:postgres@localhost:5432/{TEST_DB}"


def _postgres_available() -> bool:
    try:
        import psycopg2

        psycopg2.connect(MAINT_URL).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _postgres_available(), reason="Postgres not reachable")


def _ensure_test_db() -> None:
    import psycopg2

    conn = psycopg2.connect(MAINT_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,))
        if cur.fetchone() is None:
            cur.execute(f"CREATE DATABASE {TEST_DB}")
    conn.close()


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    _ensure_test_db()
    engine = create_async_engine(ASYNC_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as db:
        db.add(User(id="user-1", email="u1@example.com", password_hash="x"))
        db.add(User(id="user-2", email="u2@example.com", password_hash="x"))
        await db.commit()
        yield db
    await engine.dispose()


def _record(document_id: str = "doc-1", checksum: str = "content-hash-abc") -> DocumentRecord:
    return DocumentRecord(
        document=DocumentMetadata(
            document_id=document_id,
            source="uploads/xyz_paper.txt",
            content_type="text/plain",
            checksum=checksum,
            language="en",
        ),
        chunk_count=2,
    )


def _chunk_meta(document_id: str, n: int = 2) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": f"{document_id}:{i}",
            "chunk_index": i,
            "language": "en",
            "source": "uploads/xyz_paper.txt",
            "page": None,
            "token_count": 5,
            "checksum": f"chunk-ck-{i}",
            "metadata": {},
        }
        for i in range(n)
    ]


async def _save(repo: DocumentRepository, record: DocumentRecord, *, user_id: str) -> None:
    await repo.save(
        record,
        user_id=user_id,
        file_path=Path("uploads/xyz_paper.txt"),
        filename="paper.txt",
        file_size_bytes=1234,
        chunk_metadata=_chunk_meta(record.document.document_id),
    )


async def test_save_get_list_roundtrip(session: AsyncSession) -> None:
    repo = DocumentRepository(session)
    await _save(repo, _record(), user_id="user-1")
    await session.commit()

    got = await repo.get(user_id="user-1", document_id="doc-1")
    assert got.document.document_id == "doc-1"
    assert got.chunk_count == 2
    listed = await repo.list(user_id="user-1")
    assert [r.document.document_id for r in listed] == ["doc-1"]


async def test_get_is_user_scoped(session: AsyncSession) -> None:
    repo = DocumentRepository(session)
    await _save(repo, _record(), user_id="user-1")
    await session.commit()
    # user-2 must not see user-1's document
    with pytest.raises(AppError):
        await repo.get(user_id="user-2", document_id="doc-1")


async def test_delete_cascades_a_document_with_chunks(session: AsyncSession) -> None:
    """D4: deleting a document that has chunks/files must succeed (was IntegrityError)."""
    repo = DocumentRepository(session)
    await _save(repo, _record(), user_id="user-1")
    await session.commit()
    assert (await session.execute(select(DocumentChunk))).scalars().all()  # chunks exist

    await repo.delete(user_id="user-1", document_id="doc-1")
    await session.commit()

    assert (await repo.list(user_id="user-1")) == ()
    assert (await session.execute(select(DocumentChunk))).scalars().all() == []
    assert (await session.execute(select(DocumentFile))).scalars().all() == []


async def test_reupload_same_document_is_idempotent(session: AsyncSession) -> None:
    """D6: re-saving the same (content-addressed) document updates in place, not duplicates."""
    repo = DocumentRepository(session)
    await _save(repo, _record(), user_id="user-1")
    await session.commit()
    await _save(repo, _record(), user_id="user-1")  # same id + same (user, checksum)
    await session.commit()

    assert len(await repo.list(user_id="user-1")) == 1
    assert len((await session.execute(select(DocumentChunk))).scalars().all()) == 2


async def test_document_file_stores_content_checksum_not_path(session: AsyncSession) -> None:
    """D7: DocumentFile.checksum must be the file's content hash, not a hash of the path."""
    repo = DocumentRepository(session)
    await _save(repo, _record(checksum="the-real-content-hash"), user_id="user-1")
    await session.commit()

    df = (await session.execute(select(DocumentFile))).scalar_one()
    assert df.checksum == "the-real-content-hash"


# --- D5: ingestion job dual-write compensation ---------------------------------------------

def _ingestion_result() -> IngestionResult:
    doc = DocumentMetadata(
        document_id="doc-job", source="s.txt", content_type="text/plain",
        checksum="job-checksum", language="en",
    )
    chunks = tuple(
        DomainChunk(
            chunk_id=f"doc-job:{i}", document_id="doc-job", text=f"chunk {i}", language="en",
            source="s.txt", chunk_index=i, checksum=f"ck{i}", token_count=3,
        )
        for i in range(2)
    )
    return IngestionResult(document=doc, chunks=chunks)


class _FakeIngestion:
    def ingest_file(self, path: Path, *, user_id: str) -> IngestionResult:
        del path, user_id
        return _ingestion_result()


class _FakeEmbeddings:
    def embed_documents(self, texts: object) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]  # type: ignore[attr-defined]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class _FakeVectorStore:
    def __init__(self, *, fail_upsert: bool = False) -> None:
        self.fail_upsert = fail_upsert
        self.upserted = False
        self.deleted: list[str] = []

    def upsert_chunks(self, chunks: object, embeddings: object, *, user_id: str) -> None:
        if self.fail_upsert:
            raise RuntimeError("chroma down")
        self.upserted = True

    def search(self, *a: object, **k: object) -> tuple:
        return ()

    def delete_document(self, document_id: str, *, user_id: str) -> None:
        self.deleted.append(document_id)


async def _queue_job(session: AsyncSession, tmp_path: Path) -> str:
    f = tmp_path / "doc.txt"
    f.write_text("content", encoding="utf-8")  # run_ingestion_job stats the file for size
    job = await IngestionJobRepository(session).create(user_id="user-1", file_path=f)
    await session.commit()
    return job.job_id


async def test_ingestion_job_happy_path(session: AsyncSession, tmp_path: Path) -> None:
    job_id = await _queue_job(session, tmp_path)
    vs = _FakeVectorStore()

    document_id = await run_ingestion_job(
        settings=Settings(environment="test"), session=session, job_id=job_id,
        ingestion_service=_FakeIngestion(), embedding_provider=_FakeEmbeddings(), vector_store=vs,
    )

    assert document_id == "doc-job"
    assert vs.upserted
    listed = await DocumentRepository(session).list(user_id="user-1")
    assert listed[0].document.document_id == "doc-job"
    job = await IngestionJobRepository(session).get(user_id="user-1", job_id=job_id)
    assert job.status == "succeeded"


async def test_ingestion_job_compensates_vectors_on_failure(
    session: AsyncSession, tmp_path: Path
) -> None:
    """D5: if the vector write fails, no document row and no orphan vectors are left behind."""
    job_id = await _queue_job(session, tmp_path)
    vs = _FakeVectorStore(fail_upsert=True)

    with pytest.raises(RuntimeError):
        await run_ingestion_job(
            settings=Settings(environment="test"), session=session, job_id=job_id,
            ingestion_service=_FakeIngestion(), embedding_provider=_FakeEmbeddings(),
            vector_store=vs,
        )

    # DB rolled back: no document; vector cleanup attempted; job marked failed.
    assert (await session.execute(select(Document))).scalars().all() == []
    assert vs.deleted == ["doc-job"]
    job = await IngestionJobRepository(session).get(user_id="user-1", job_id=job_id)
    assert job.status == "failed"


async def test_job_lifecycle(session: AsyncSession) -> None:
    # A document must exist first — mark_succeeded sets a FK to documents.id.
    await _save(DocumentRepository(session), _record(), user_id="user-1")
    jobs = IngestionJobRepository(session)
    job = await jobs.create(user_id="user-1", file_path=Path("uploads/xyz_paper.txt"))
    await session.commit()
    assert job.status == "queued"

    await jobs.mark_running(job.job_id)
    await jobs.mark_succeeded(job_id=job.job_id, document_id="doc-1")
    await session.commit()
    fetched = await jobs.get(user_id="user-1", job_id=job.job_id)
    assert fetched.status == "succeeded"
    assert fetched.document_id == "doc-1"
