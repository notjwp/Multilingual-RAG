"""Deterministic text chunking utilities."""

from __future__ import annotations

import re
from collections.abc import Iterable

from multilingual_rag.core.models import DocumentChunk, DocumentSection
from multilingual_rag.ingestion.language import LanguageDetector
from multilingual_rag.ingestion.service_utils import checksum_text

TOKEN_PATTERN = re.compile(r"\S+")


class TextChunker:
    """Split document sections into overlapping token windows."""

    def __init__(self, chunk_size_tokens: int, chunk_overlap_tokens: int) -> None:
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be greater than zero")
        if chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be greater than or equal to zero")
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

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
            token_spans = list(iter_token_spans(normalized_text))

            for start in range(0, len(token_spans), self._step_size):
                window = token_spans[start : start + self.chunk_size_tokens]
                if not window:
                    continue

                chunk_text = normalized_text[window[0][0] : window[-1][1]].strip()
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

                if start + self.chunk_size_tokens >= len(token_spans):
                    break

        return tuple(chunks)

    @property
    def _step_size(self) -> int:
        return self.chunk_size_tokens - self.chunk_overlap_tokens


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving readable text boundaries."""
    return re.sub(r"\s+", " ", text).strip()


def iter_token_spans(text: str) -> Iterable[tuple[int, int]]:
    """Yield approximate token spans based on non-whitespace runs."""
    for match in TOKEN_PATTERN.finditer(text):
        yield match.span()

