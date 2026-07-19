"""Detect romanized Hindi — decides *whether* to transliterate a Latin-script query.

Score-based routing proved unreliable at scale (the raw romanized search finds enough
high-cosine noise to look confident). Two detectors exist, chosen by ``TRANSLITERATION_DETECTOR``:

- **word-list** (default): a cheap linguistic check — romanized Hindi is saturated with function
  words that essentially never appear in English (`kya`, `hai`, `kaun`, `kahan`, `nahi`, …). We
  exclude tokens that collide with English (`is`, `to`, `me`, `the`, `par`, `ka`, `ki`) to keep
  **precision** high. Scores ~98.3% recall / 0% FP on the eval.
- **muril** (opt-in): a MuRIL feature + LogisticRegression head (`muril.py`,
  `scripts/train_romanized_detector.py`). No measured eval gain over the word list, but generalizes
  better to real-world spelling variants. Falls back to the word list on any failure (missing
  artifact, no torch), so selecting it can never hard-break detection.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from multilingual_rag.transliteration.script import is_latin_script

logger = logging.getLogger(__name__)

# Repo-root/data/models/... — detect.py is src/multilingual_rag/transliteration/detect.py.
_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "romanized_hi_detector.joblib"
)

# The MuRIL path self-disables after a failure so it doesn't retry the model on every query.
_state = {"degraded": False}

# Distinctly-Hindi romanized tokens with no common English collision. One is enough signal.
# Each concept lists both the natural-typing form and the IAST-strip form our romanizer emits
# (kaun/kauna, kitne/kitane, mein/mem, …), so detection fires on real input AND on the eval.
# Deliberately EXCLUDES English collisions (the/log/mere/is/to/me/par/ka/ki) — a false positive
# would mis-transliterate a real English query, so precision is prioritized.
_HINDI_MARKERS: frozenset[str] = frozenset(
    {
        # question words
        "kya", "kyaa", "kaun", "kauna", "kaunsa", "kaunasa", "kaunsi", "kahan", "kahaan", "kaha",
        "kitne", "kitane", "kitna", "kitana", "kitni", "kitani", "kab", "kaba",
        "kaise", "kaisa", "kaisi", "kyun", "kyu", "kyon", "kyom", "kyunki", "kyonki",
        "kis", "kisa", "kisne", "kisane", "kiska", "kisaka", "kiski", "kispe",
        # verbs / copulas
        "hai", "hain", "tha", "thi", "hoga", "hogi", "honge", "hota", "hoti", "hote",
        "raha", "rahi", "rahe", "karta", "karti", "karte", "karna", "kiya", "gaya", "gayi",
        # pronouns / particles distinctive to Hindi
        "mera", "meraa", "meri", "tera", "teri", "tumhara", "hamara", "hamari",
        "uska", "usaka", "uski", "unka", "unaka", "unki", "iska", "isaka", "iski",
        "mujhe", "tumhe", "hume", "humein", "aap", "aapka",
        "nahi", "nahin", "nahim", "haan", "kuch", "kuchh", "bahut", "thoda", "sabhi",
        "wala", "wale", "wali", "yahan", "wahan", "abhi", "phir", "lekin", "magar",
        "mein", "mem",
        # content words that are unmistakably Hindi
        "naam", "nama", "paani", "khana", "khata", "ghar", "aadmi", "aurat", "bharat", "bharata",
        "duniya", "rajdhani", "rajadhani", "shahar", "desh", "aur", "aura",
    }
)

_WORD_RE = re.compile(r"[a-z]+")


def _wordlist_match(text: str) -> bool:
    """True when ``text`` contains at least one distinctly-Hindi romanized token."""
    return any(word in _HINDI_MARKERS for word in _WORD_RE.findall(text.lower()))


class _MurilDetector:
    """MuRIL embedding → LogisticRegression head → boolean, at a tuned threshold."""

    def __init__(self, clf: Any, threshold: float, extractor: Any) -> None:
        self._clf = clf
        self._threshold = threshold
        self._extractor = extractor

    def predict(self, text: str) -> bool:
        vector = self._extractor.embed([text])  # (1, 768)
        proba = float(self._clf.predict_proba(vector)[0, 1])
        return proba >= self._threshold


@lru_cache(maxsize=1)
def _load_detector() -> _MurilDetector | None:
    """Load the trained MuRIL detector once, or None on any failure (then word-list is used).

    Returns None when the artifact is missing or torch/transformers/sklearn/joblib can't import
    (e.g. offline, or a fresh checkout that never trained the head) — so the feature degrades to
    the word list rather than erroring.
    """
    try:
        import joblib  # type: ignore[import-untyped]

        from multilingual_rag.transliteration.muril import MurilFeatureExtractor

        payload = joblib.load(_ARTIFACT_PATH)
        # CPU: the detector co-resides with bge-m3 on the query path; a tiny forward pass on CPU
        # avoids competing for a small GPU. Training uses its own (GPU) extractor.
        extractor = MurilFeatureExtractor(device="cpu")
        return _MurilDetector(payload["clf"], float(payload["threshold"]), extractor)
    except Exception:
        logger.warning("MuRIL detector unavailable; using word-list detection", exc_info=True)
        return None


def is_romanized_indic(
    text: str, languages: tuple[str, ...], *, detector: str = "word-list"
) -> bool:
    """True when ``text`` looks like romanized Hindi that should be transliterated.

    Only Hindi is implemented; kn/te would need their own detectors. Requires Latin script (a
    native-Devanagari query is already fine). ``detector`` selects the classifier: ``"word-list"``
    (default) or ``"muril"`` (opt-in, falls back to the word list on any failure).
    """
    if "hi" not in languages or not is_latin_script(text):
        return False
    if detector == "muril" and not _state["degraded"]:
        model = _load_detector()
        if model is not None:
            try:
                return model.predict(text)
            except Exception:
                logger.warning("MuRIL detection failed; word-list fallback", exc_info=True)
                _state["degraded"] = True
    return _wordlist_match(text)
