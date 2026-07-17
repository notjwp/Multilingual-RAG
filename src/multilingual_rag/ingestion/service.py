"""High-level ingestion orchestration."""

from __future__ import annotations

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import DocumentMetadata, IngestionResult, LoadedDocument
from multilingual_rag.ingestion.chunker import TextChunker, normalize_text
from multilingual_rag.ingestion.language import LanguageDetector
from multilingual_rag.ingestion.loaders import DocumentLoader
from multilingual_rag.ingestion.service_utils import checksum_text
from multilingual_rag.ingestion.tokenizer import BgeM3Tokenizer


class IngestionService:
    """Parse documents, detect language, and produce chunks."""

    def __init__(
        self,
        settings: Settings,
        *,
        loader: DocumentLoader | None = None,
        language_detector: LanguageDetector | None = None,
        chunker: TextChunker | None = None,
    ) -> None:
        self.settings = settings
        self.loader = loader or DocumentLoader()
        self.language_detector = language_detector or LanguageDetector()
        self.chunker = chunker or TextChunker(
            BgeM3Tokenizer(),
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        )

    def ingest_file(self, path: Path) -> IngestionResult:
        """Load a file from disk and convert it into indexable chunks."""
        loaded_document = self.loader.load(path)
        return self.ingest_loaded_document(loaded_document)

    def ingest_loaded_document(self, loaded_document: LoadedDocument) -> IngestionResult:
        """Convert a loaded document into metadata and chunks."""
        document_text = normalize_text(
            " ".join(section.text for section in loaded_document.sections if section.text.strip())
        )
        if not document_text:
            raise AppError(
                "Document does not contain extractable text.",
                code="empty_document",
                status_code=400,
            )

        checksum = checksum_text(document_text)
        document_id = str(uuid5(NAMESPACE_URL, f"{loaded_document.source_path}:{checksum}"))
        language = self.language_detector.detect(document_text)

        metadata = DocumentMetadata(
            document_id=document_id,
            source=str(loaded_document.source_path),
            content_type=loaded_document.content_type,
            checksum=checksum,
            language=language,
            extra=loaded_document.metadata,
        )
        chunks = self.chunker.chunk_sections(
            document_id=document_id,
            source=metadata.source,
            sections=loaded_document.sections,
            language_detector=self.language_detector,
            default_language=language,
        )

        if not chunks:
            raise AppError(
                "Document did not produce any chunks.",
                code="empty_document_chunks",
                status_code=400,
            )

        return IngestionResult(document=metadata, chunks=chunks)

