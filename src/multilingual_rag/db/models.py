"""Database ORM models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from multilingual_rag.db.base import Base


class IngestionStatus(StrEnum):
    """Document ingestion lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class User(Base):
    """Application user."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        nullable=False,
    )

    documents: Mapped[list[Document]] = relationship(back_populates="user")


class Document(Base):
    """User-owned document metadata."""

    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_user_status", "user_id", "ingestion_status"),
        # One document per (user, content) — the content-addressed dedup safety net.
        UniqueConstraint("user_id", "checksum", name="uq_documents_user_checksum"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="unknown")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ingestion_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=IngestionStatus.QUEUED,
    )
    extra: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        nullable=False,
    )

    user: Mapped[User] = relationship(back_populates="documents")
    file: Mapped[DocumentFile | None] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentFile(Base):
    """Stored original file metadata."""

    __tablename__ = "document_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    path: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum: Mapped[str] = mapped_column(String(128), nullable=False)

    document: Mapped[Document] = relationship(back_populates="file")


class IngestionJob(Base):
    """Async ingestion job status."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    document_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=IngestionStatus.QUEUED)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        onupdate=lambda: datetime.now(tz=UTC),
        nullable=False,
    )


class ChatSession(Base):
    """User-owned chat session placeholder for milestone 14."""

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        nullable=False,
    )


class Message(Base):
    """Persisted chat message placeholder for milestone 14."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        nullable=False,
    )


class MessageCitation(Base):
    """Assistant message citation placeholder for milestone 14."""

    __tablename__ = "message_citations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    message_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)


class DocumentChunk(Base):
    """Chunk metadata mirrored from ChromaDB for traceability."""

    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[str] = mapped_column(String(128), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    chunk_metadata: Mapped[dict[str, object]] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
