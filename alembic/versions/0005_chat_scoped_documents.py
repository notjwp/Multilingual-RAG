"""per-chat document scoping: add session_id to documents + ingestion_jobs

Revision ID: 0005_chat_scoped_documents
Revises: 0004_chat_fk_cascade
Create Date: 2026-07-24

M18 scopes documents to a single chat instead of the whole user. Add a nullable ``session_id``
FK (ON DELETE CASCADE, so deleting a chat removes its documents) to ``documents`` and
``ingestion_jobs``, and widen the dedup unique constraint from (user_id, checksum) to
(user_id, session_id, checksum) so the same file can live in different chats. Existing rows keep
``session_id = NULL`` (user-wide, orphaned under the new model — re-upload inside a chat).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_chat_scoped_documents"
down_revision: str | None = "0004_chat_fk_cascade"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("session_id", sa.String(length=36), nullable=True))
    op.add_column("ingestion_jobs", sa.Column("session_id", sa.String(length=36), nullable=True))
    op.create_index("ix_documents_session_id", "documents", ["session_id"])
    op.create_foreign_key(
        "documents_session_id_fkey",
        "documents",
        "chat_sessions",
        ["session_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "ingestion_jobs_session_id_fkey",
        "ingestion_jobs",
        "chat_sessions",
        ["session_id"],
        ["id"],
        ondelete="CASCADE",
    )
    # Widen dedup to be chat-scoped: the same content can now be a distinct document per chat.
    op.drop_constraint("uq_documents_user_checksum", "documents", type_="unique")
    op.create_unique_constraint(
        "uq_documents_user_session_checksum",
        "documents",
        ["user_id", "session_id", "checksum"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_documents_user_session_checksum", "documents", type_="unique")
    op.create_unique_constraint(
        "uq_documents_user_checksum", "documents", ["user_id", "checksum"]
    )
    op.drop_constraint("ingestion_jobs_session_id_fkey", "ingestion_jobs", type_="foreignkey")
    op.drop_constraint("documents_session_id_fkey", "documents", type_="foreignkey")
    op.drop_index("ix_documents_session_id", table_name="documents")
    op.drop_column("ingestion_jobs", "session_id")
    op.drop_column("documents", "session_id")
