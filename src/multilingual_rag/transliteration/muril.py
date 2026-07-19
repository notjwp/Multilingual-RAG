"""MuRIL as a frozen feature extractor for romanized-Hindi detection.

MuRIL (`google/muril-base-cased`) is a BERT trained on 17 Indian languages **and their
transliterated counterparts**, so its embeddings understand romanized Hindi — unlike bge-m3 or a
code model. Here it is used *frozen* (no fine-tune): mean-pooled token embeddings feed a tiny
LogisticRegression head (see `scripts/train_romanized_detector.py`). Loaded lazily and once,
revision-pinned, mirroring `bge_embeddings.py` / `indicxlit.py`.

This is opt-in (`TRANSLITERATION_DETECTOR=muril`); the default detector is the word list, so this
model is never loaded unless explicitly selected.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np

MODEL_NAME = "google/muril-base-cased"
# Pinned so the extracted features (and thus the trained head) stay reproducible.
MODEL_REVISION = "afd9f36c7923d54e97903922ff1b260d091d202f"
EMBED_DIM = 768


@lru_cache(maxsize=1)
def _load(device: str | None) -> tuple[Any, Any, str]:
    """Load MuRIL's tokenizer + model once per (process, device)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).to(resolved)
    model.eval()
    return tokenizer, model, resolved


class MurilFeatureExtractor:
    """Turn text into a 768-dim mean-pooled MuRIL embedding (frozen; no gradients)."""

    def __init__(self, *, device: str | None = None, batch_size: int = 32) -> None:
        self.device = device
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Mean-pool the last hidden state over the attention mask → ``(n, 768)`` float32.

        Batched so a large training corpus (~3.5k texts) doesn't exhaust a small GPU.
        """
        import torch

        items = list(texts)
        if not items:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

        tokenizer, model, device = _load(self.device)
        vectors: list[np.ndarray] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                hidden = model(**encoded).last_hidden_state  # (b, seq, 768)
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (b, seq, 1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts  # (b, 768)
            vectors.append(pooled.cpu().float().numpy())
        return np.concatenate(vectors, axis=0)
