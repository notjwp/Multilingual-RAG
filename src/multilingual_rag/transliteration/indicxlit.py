"""Neural roman->native transliteration via a local Gemma-3 char-model (free, offline).

AI4Bharat's IndicXlit needs ``fairseq``, which won't build on Python 3.13 (the spike's
blocker). This uses ``psidharth567/indic-xlit-50M`` instead — a ~52M-param Gemma-3 fine-tune
with a character tokenizer that loads with ``transformers`` (already a dep via
sentence-transformers) and runs on CPU/GPU. Prompt format is
``[BOS][LANG]{source}[SEP]{target}[EOS]``; we generate the target greedily and trim the
model's occasional trailing-token / danda artifacts.

Robustness: the model is revision-pinned, loaded lazily and once. If it can't load or a call
fails (offline, a torch/transformers mismatch, the checkpoint vanished), the adapter falls back
to the rule-based transliterator so the query path always has a working local transliterator.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from multilingual_rag.transliteration.base import Transliterator
from multilingual_rag.transliteration.rule_based import SanscriptTransliterator

logger = logging.getLogger(__name__)

MODEL_ID = "psidharth567/indic-xlit-50M"
# Pinned so the transliteration behaviour is reproducible and can't change under us.
MODEL_REVISION = "0754ae36d10d31642a636c8e0386378ad7119777"

# language -> the model's target-script tag. Hindi ships; kn/te wired for when validated.
_LANG_TAG: dict[str, str] = {"hi": "[HINDI]", "kn": "[KANNADA]", "te": "[TELUGU]"}

# Devanagari danda / double-danda the model sometimes appends as sentence punctuation.
_TRAILING_PUNCT = "।॥ \t"


@lru_cache(maxsize=1)
def _load(device: str | None) -> tuple[Any, dict[str, int], dict[int, str], str]:
    """Load model + char vocab once. Returns (model, token->id, id->token, device)."""
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM

    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")

    vocab_path = hf_hub_download(
        MODEL_ID, "char_tokenizer/vocab.json", revision=MODEL_REVISION
    )
    with open(vocab_path, encoding="utf-8") as handle:
        vocab: dict[str, int] = json.load(handle)
    id_to_token = {index: token for token, index in vocab.items()}

    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        config=config,
        dtype=torch.bfloat16 if resolved == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(resolved)  # type: ignore[arg-type]
    model.eval()

    # Gemma wraps embeddings in a ScaledWordEmbedding that generation mishandles here; swap in
    # a plain Embedding with the same weights (per the model card's required fix).
    embedding = model.model.embed_tokens
    if "ScaledWordEmbedding" in str(type(embedding)):
        plain = nn.Embedding(
            embedding.num_embeddings,
            embedding.embedding_dim,
            padding_idx=getattr(embedding, "padding_idx", None),
            device=next(embedding.parameters()).device,
            dtype=next(embedding.parameters()).dtype,
        )
        plain.weight.data = embedding.weight.data.clone()
        model.model.embed_tokens = plain

    return model, vocab, id_to_token, resolved


def _clean(text: str) -> str:
    """Trim the model's artifacts: a trailing danda and a duplicated final token."""
    text = text.strip().strip(_TRAILING_PUNCT).strip()
    words = text.split()
    if len(words) >= 2 and words[-1] == words[-2]:
        words = words[:-1]
    return " ".join(words)


class IndicXlitTransliterator:
    """Neural transliterator with a rule-based safety net. Satisfies ``Transliterator``."""

    def __init__(
        self,
        *,
        device: str | None = None,
        max_new_tokens: int = 64,
        fallback: Transliterator | None = None,
    ) -> None:
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._fallback = fallback or SanscriptTransliterator()
        self._degraded = False  # flips true once the model fails, to stop retrying per call

    def transliterate(self, text: str, *, target_language: str) -> str:
        tag = _LANG_TAG.get(target_language)
        if tag is None or not text.strip():
            return text
        if self._degraded:
            return self._fallback.transliterate(text, target_language=target_language)
        try:
            return self._generate(text, tag) or text
        except Exception:
            logger.warning(
                "IndicXlit transliteration failed; using rule-based fallback", exc_info=True
            )
            self._degraded = True
            return self._fallback.transliterate(text, target_language=target_language)

    def _generate(self, text: str, tag: str) -> str:
        import torch

        model, vocab, id_to_token, device = _load(self._device)
        bos, sep, eos, pad, unk = (
            vocab["[BOS]"], vocab["[SEP]"], vocab["[EOS]"], vocab["[PAD]"], vocab["[UNK]"]
        )
        ids = [bos, vocab[tag]] + [vocab.get(char, unk) for char in text] + [sep]
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=eos,
                pad_token_id=pad,
            )
        generated = output[0].tolist()[input_ids.shape[1] :]
        chars: list[str] = []
        for token_id in generated:
            if token_id == eos:
                break
            chars.append(id_to_token.get(token_id, ""))
        return _clean("".join(chars))
