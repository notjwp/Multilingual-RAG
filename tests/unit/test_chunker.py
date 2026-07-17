from multilingual_rag.core.models import DocumentSection
from multilingual_rag.ingestion.chunker import TextChunker, normalize_text
from multilingual_rag.ingestion.language import LanguageDetector


class FakeWordTokenizer:
    """Words-as-tokens tokenizer that round-trips — deterministic, no model, fast.

    Lets the windowing logic be tested exactly, independent of bge-m3.
    """

    def __init__(self) -> None:
        self._vocab: list[str] = []
        self._index: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            if word not in self._index:
                self._index[word] = len(self._vocab)
                self._vocab.append(word)
            ids.append(self._index[word])
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self._vocab[token_id] for token_id in token_ids)


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  hello\n\nworld\t ") == "hello world"


def test_chunker_creates_overlapping_chunks() -> None:
    chunker = TextChunker(FakeWordTokenizer(), chunk_size_tokens=4, chunk_overlap_tokens=1)
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
    # token_count is now real tokens (window length), and no chunk exceeds the size.
    assert chunks[0].token_count == 4
    assert all(chunk.token_count <= 4 for chunk in chunks)


def test_chunker_single_window_for_short_text() -> None:
    chunker = TextChunker(FakeWordTokenizer(), chunk_size_tokens=10, chunk_overlap_tokens=2)
    section = DocumentSection(text="short text here")

    chunks = chunker.chunk_sections(
        document_id="doc-1",
        source="s.txt",
        sections=(section,),
        language_detector=LanguageDetector(min_text_length=0),
        default_language="en",
    )

    assert len(chunks) == 1
    assert chunks[0].text == "short text here"
    assert chunks[0].token_count == 3


def test_chunker_rejects_invalid_overlap() -> None:
    try:
        TextChunker(FakeWordTokenizer(), chunk_size_tokens=10, chunk_overlap_tokens=10)
    except ValueError as exc:
        assert "chunk_overlap_tokens" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
