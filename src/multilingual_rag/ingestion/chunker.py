"""Deterministic text chunking over the embedding model's own tokens."""

from __future__ import annotations

import re
from collections.abc import Iterable

from multilingual_rag.core.models import DocumentChunk, DocumentSection
from multilingual_rag.ingestion.language import LanguageDetector
from multilingual_rag.ingestion.service_utils import checksum_text
from multilingual_rag.ingestion.tokenizer import Tokenizer


class TextChunker:
    """Split document sections into overlapping windows of real (sub-word) tokens.

    Windowing over the embedding model's tokenizer — not whitespace — is what makes CJK/Thai
    chunk correctly: those scripts have no inter-word spaces, so ``\\S+`` collapsed a whole
    document into one oversized chunk.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_size_tokens: int,
        chunk_overlap_tokens: int,
    ) -> None:
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be greater than zero")
        if chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be greater than or equal to zero")
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

        self.tokenizer = tokenizer
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    def chunk_sections(
        self,
        *,
        document_id: str,
        source: str,
        sections: Iterable[DocumentSection],
        language_detector: LanguageDetector,
        default_language: str,
    ) -> tuple[DocumentChunk, ...]:
        """Chunk document sections and attach metadata for downstream indexing."""
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for section in sections:
            normalized_text = normalize_text(section.text)
            if not normalized_text:
                continue

            section_language = language_detector.detect(normalized_text, default=default_language)
            token_ids = self.tokenizer.encode(normalized_text)

            for start in range(0, len(token_ids), self._step_size):
                window = token_ids[start : start + self.chunk_size_tokens]
                if not window:
                    continue

                chunk_text = self.tokenizer.decode(window).strip()
                if not chunk_text:
                    continue

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}:{chunk_index}",
                        document_id=document_id,
                        text=chunk_text,
                        language=section_language,
                        source=source,
                        chunk_index=chunk_index,
                        checksum=checksum_text(chunk_text),
                        page=section.page,
                        token_count=len(window),
                        metadata={
                            **section.metadata,
                            "section_index": section.section_index,
                        },
                    )
                )
                chunk_index += 1

                if start + self.chunk_size_tokens >= len(token_ids):
                    break

        return tuple(chunks)

    @property
    def _step_size(self) -> int:
        return self.chunk_size_tokens - self.chunk_overlap_tokens


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving readable text boundaries."""
    return re.sub(r"\s+", " ", text).strip()
