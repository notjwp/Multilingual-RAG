from multilingual_rag.core.models import DocumentSection
from multilingual_rag.ingestion.chunker import TextChunker, normalize_text
from multilingual_rag.ingestion.language import LanguageDetector


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  hello\n\nworld\t ") == "hello world"


def test_chunker_creates_overlapping_chunks() -> None:
    chunker = TextChunker(chunk_size_tokens=4, chunk_overlap_tokens=1)
    detector = LanguageDetector(min_text_length=0)
    section = DocumentSection(
        text="one two three four five six seven",
        page=2,
        section_index=3,
    )

    chunks = chunker.chunk_sections(
        document_id="doc-1",
        source="sample.txt",
        sections=(section,),
        language_detector=detector,
        default_language="en",
    )

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
    ]
    assert chunks[0].chunk_id == "doc-1:0"
    assert chunks[0].page == 2
    assert chunks[0].metadata["section_index"] == 3


def test_chunker_rejects_invalid_overlap() -> None:
    try:
        TextChunker(chunk_size_tokens=10, chunk_overlap_tokens=10)
    except ValueError as exc:
        assert "chunk_overlap_tokens" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

