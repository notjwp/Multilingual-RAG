"""Tokenizer contract for chunking in the embedding model's own tokens.

Chunk sizes only mean something if they are counted in the tokens the embedder actually uses.
Whitespace splitting (the old ``\\S+``) collapses Chinese/Thai — which have no inter-word
spaces — into one giant "token", so a whole CJK document became a single chunk that then
overran the model's input limit (M0 measured ~96% of a Chinese article silently dropped).

``BgeM3Tokenizer`` loads only the tokenizer (a few MB of sentencepiece files), not the 2.2 GB
model, so ingestion stays light.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Pinned to match embeddings/bge_embeddings.py so chunk tokens == embedded tokens.
TOKENIZER_NAME = "BAAI/bge-m3"
TOKENIZER_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


class Tokenizer(Protocol):
    """Encodes text to token ids and back, for windowing during chunking."""

    def encode(self, text: str) -> list[int]:
        """Return token ids for ``text`` (no special tokens)."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Reconstruct text from token ids."""
        ...


@lru_cache(maxsize=1)
def _load_tokenizer() -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer

    return cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION),
    )


class BgeM3Tokenizer:
    """The bge-m3 (XLM-RoBERTa) tokenizer. Satisfies the ``Tokenizer`` protocol."""

    def encode(self, text: str) -> list[int]:
        return list(_load_tokenizer().encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return str(_load_tokenizer().decode(token_ids, skip_special_tokens=True))
