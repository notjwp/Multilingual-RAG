"""C1 headline proof: a long Chinese document chunks into many pieces, not one blob.

Uses the real bge-m3 tokenizer, so it is opt-in (RUN_MODEL_TESTS=1). Under the old ``\\S+``
chunker this document was a single chunk that overran the embedder; here it must split.
"""

import os
from importlib.util import find_spec

import pytest

from multilingual_rag.core.models import DocumentSection
from multilingual_rag.ingestion.chunker import TextChunker
from multilingual_rag.ingestion.language import LanguageDetector

_ENABLED = find_spec("transformers") is not None and os.environ.get("RUN_MODEL_TESTS")

pytestmark = pytest.mark.skipif(
    not _ENABLED,
    reason="set RUN_MODEL_TESTS=1 with transformers installed to run the CJK tokenizer check",
)

# ~1500 chars of Chinese with no inter-word spaces — the case \S+ collapsed to one token.
_CHINESE = (
    "黑豹队的防守只丢了308分，在联赛中排名第六，同时也以24次拦截领先国家橄榄球联盟，"
    "并且四次入选职业碗。职业碗防守截锋卡万·肖特以11次擒杀领先全队，同时迫使三次掉球并"
    "恢复两次。同为锋线球员的马里奥·阿迪森贡献了六个半擒杀。黑豹队的锋线还拥有老将防守端"
    "锋贾里德·艾伦，他是五次职业碗球员，也是国家橄榄球联盟现役生涯擒杀领先者，共有136次。"
) * 8


def test_long_chinese_document_splits_into_many_chunks() -> None:
    from multilingual_rag.ingestion.tokenizer import BgeM3Tokenizer

    chunker = TextChunker(BgeM3Tokenizer(), chunk_size_tokens=128, chunk_overlap_tokens=16)
    chunks = chunker.chunk_sections(
        document_id="zh-doc",
        source="zh.txt",
        sections=(DocumentSection(text=_CHINESE),),
        language_detector=LanguageDetector(),
        default_language="zh",
    )

    # The whole point: not one truncated blob.
    assert len(chunks) > 1
    # Every chunk respects the token budget and is valid, non-empty, undamaged text.
    assert all(0 < chunk.token_count <= 128 for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)
    assert all("�" not in chunk.text for chunk in chunks)
    # langdetect returns BCP-47-ish codes (zh-cn / zh-tw), not bare "zh".
    assert all(chunk.language.startswith("zh") for chunk in chunks)
