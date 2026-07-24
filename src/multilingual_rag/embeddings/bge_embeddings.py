"""Local bge-m3 embedding adapter (free) for offline evaluation.

Selected over multilingual-e5-large in M0 (see ``docs/m0/report.md``): stronger cross-lingual
retention and — critically — **no query/passage prefixes** (unlike e5). Runs locally via
sentence-transformers, so evaluation costs nothing. Built here for the eval harness; the
production ``/v1/query`` swap and Chroma re-index are Phase C.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from multilingual_rag.embeddings.base import EmbeddingVector

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
# Pinned to the exact weights M0 measured (docs/m0/report.md), for reproducible baselines.
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_DIM = 1024


@lru_cache(maxsize=1)
def _load_model(device: str | None) -> SentenceTransformer:
    """Load the ~2.2 GB model once per (process, device) — never per call."""
    from multilingual_rag.core.config import get_settings

    if get_settings().hf_hub_offline:
        # The model is cached locally; skip Hugging Face Hub network checks (~6s/load). Set before
        # importing sentence-transformers so huggingface_hub reads these at its import.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from sentence_transformers import SentenceTransformer

    return cast(
        "SentenceTransformer",
        SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device=device),
    )


class BgeM3EmbeddingProvider:
    """Embed text with bge-m3. Satisfies the ``EmbeddingProvider`` protocol."""

    def __init__(self, *, batch_size: int = 16, device: str | None = None) -> None:
        self.batch_size = batch_size
        self.device = device

    def embed_documents(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Embed document texts. bge-m3 takes raw text — no ``passage:`` prefix."""
        return self._embed(list(texts))

    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed one query. bge-m3 takes raw text — no ``query:`` prefix."""
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        vectors = cast(
            Any,
            _load_model(self.device).encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
        )
        return [[float(value) for value in vector] for vector in vectors]
