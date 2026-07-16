"""File-backed document metadata store."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import status
from pydantic import TypeAdapter

from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import DocumentRecord


class DocumentStore:
    """Persist document records to a local JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, record: DocumentRecord) -> None:
        """Insert or replace one document record."""
        records = self._read_records()
        records[record.document.document_id] = record
        self._write_records(records)

    def get(self, document_id: str) -> DocumentRecord:
        """Return a document record by ID."""
        records = self._read_records()
        try:
            return records[document_id]
        except KeyError as exc:
            raise AppError(
                f"Document not found: {document_id}",
                code="document_not_found",
                status_code=status.HTTP_404_NOT_FOUND,
            ) from exc

    def list(self) -> tuple[DocumentRecord, ...]:
        """Return all stored document records."""
        return tuple(self._read_records().values())

    def delete(self, document_id: str) -> DocumentRecord:
        """Delete and return a document record."""
        records = self._read_records()
        try:
            record = records.pop(document_id)
        except KeyError as exc:
            raise AppError(
                f"Document not found: {document_id}",
                code="document_not_found",
                status_code=status.HTTP_404_NOT_FOUND,
            ) from exc

        self._write_records(records)
        return record

    def _read_records(self) -> dict[str, DocumentRecord]:
        if not self.path.exists():
            return {}

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AppError(
                f"Document store is not valid JSON: {self.path}",
                code="invalid_document_store",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

        adapter = TypeAdapter(dict[str, DocumentRecord])
        return adapter.validate_python(raw)

    def _write_records(self, records: dict[str, DocumentRecord]) -> None:
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        payload = {
            document_id: record.model_dump(mode="json")
            for document_id, record in sorted(records.items())
        }
        temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary_path.replace(self.path)

