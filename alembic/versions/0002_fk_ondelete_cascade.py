"""fk ondelete cascade so documents can be deleted

Revision ID: 0002_fk_ondelete_cascade
Revises: 0001_initial_schema
Create Date: 2026-07-18

The initial schema created child FKs to documents with no ON DELETE rule, so deleting a document
that had chunks/files raised a ForeignKeyViolationError (the DELETE endpoint was broken). Recreate
those FKs with ON DELETE CASCADE (children go with the parent) and SET NULL for the job's optional
document reference (preserve job history).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002_fk_ondelete_cascade"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CASCADE = (("document_files", "CASCADE"), ("document_chunks", "CASCADE"),
            ("message_citations", "CASCADE"), ("ingestion_jobs", "SET NULL"))


def upgrade() -> None:
    for table, rule in _CASCADE:
        name = f"{table}_document_id_fkey"
        op.drop_constraint(name, table, type_="foreignkey")
        op.create_foreign_key(
            name, table, "documents", ["document_id"], ["id"], ondelete=rule
        )


def downgrade() -> None:
    for table, _ in _CASCADE:
        name = f"{table}_document_id_fkey"
        op.drop_constraint(name, table, type_="foreignkey")
        op.create_foreign_key(name, table, "documents", ["document_id"], ["id"])
