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

import asyncio
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from multilingual_rag.transliteration.script import is_latin_script

logger = logging.getLogger(__name__)

# Minimum googletrans language-detection confidence to trust a detected language.
_DETECT_CONFIDENCE = 0.5

# Repo-root/data/models/... — detect.py is src/multilingual_rag/transliteration/detect.py.
_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "romanized_indic_detector.joblib"
)
# Languages the MuRIL head can predict (besides "other").
_INDIC_LANGS = frozenset({"hi", "kn", "te"})

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
    """MuRIL embedding → multinomial LR head → the romanized language, or None."""

    def __init__(self, clf: Any, threshold: float, classes: list[str], extractor: Any) -> None:
        self._clf = clf
        self._threshold = threshold
        self._classes = classes
        self._extractor = extractor

    def predict_language(self, text: str) -> str | None:
        proba = self._clf.predict_proba(self._extractor.embed([text]))[0]  # (n_classes,)
        best = int(proba.argmax())
        language, confidence = self._classes[best], float(proba[best])
        if language in _INDIC_LANGS and confidence >= self._threshold:
            return language
        return None  # "other", or an Indic guess too weak to trust


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
        return _MurilDetector(
            payload["clf"], float(payload["threshold"]), list(payload["classes"]), extractor
        )
    except Exception:
        logger.warning("MuRIL detector unavailable; using word-list detection", exc_info=True)
        return None


def _google_detect(text: str) -> tuple[str, float] | None:
    """Detect the language of romanized ``text`` via googletrans → (lang, confidence), or None.

    Google Translate recognizes romanized Indian languages (tested: kn/te at high confidence),
    which no local list does — this is what makes multi-language (hi/kn/te) detection possible
    without per-language training data. Network call; returns None on any failure.
    """

    async def _run() -> Any:
        import httpx
        from googletrans import Translator  # type: ignore[import-untyped]

        async with Translator(raise_exception=True, timeout=httpx.Timeout(5.0)) as translator:
            return await translator.detect(text)

    try:
        detected = asyncio.run(_run())
    except Exception:
        logger.warning("googletrans language detection failed", exc_info=True)
        return None

    lang, confidence = detected.lang, detected.confidence
    if isinstance(lang, list):  # googletrans may return lists for ambiguous input
        lang = lang[0] if lang else None
    if isinstance(confidence, list):
        confidence = confidence[0] if confidence else 0.0
    if not lang:
        return None
    return str(lang), float(confidence) if confidence is not None else 0.0


def detect_target_language(
    text: str, languages: tuple[str, ...], *, detector: str = "word-list"
) -> str | None:
    """Return which configured Indic language ``text`` is romanized in, or None to skip.

    Requires Latin script (a native-script query is already fine). ``detector``:
    - ``"word-list"``: Hindi only — return ``"hi"`` or None.
    - ``"muril"``: local multi-class MuRIL head — returns hi/kn/te (or None for "other"), no
      network; falls back to the Hindi word list if the model can't load.
    - ``"google"``: googletrans detects hi/kn/te via a network call; same word-list safety net.
    Both multi-language detectors only return a language that is also in ``languages``.
    """
    if not is_latin_script(text):
        return None

    if detector == "google":
        result = _google_detect(text)
        if result is not None:
            lang, confidence = result
            if lang in languages and confidence >= _DETECT_CONFIDENCE:
                return lang
        # Unavailable / unconfident / other-language → Hindi word-list safety net (English → None).
        return "hi" if ("hi" in languages and _wordlist_match(text)) else None

    if detector == "muril" and not _state["degraded"]:
        # Local multi-class MuRIL head: identifies hi/kn/te (or "other" → None), no network.
        model = _load_detector()
        if model is not None:
            try:
                detected = model.predict_language(text)
                return detected if (detected is not None and detected in languages) else None
            except Exception:
                logger.warning("MuRIL detection failed; word-list fallback", exc_info=True)
                _state["degraded"] = True
        return "hi" if ("hi" in languages and _wordlist_match(text)) else None

    # word-list detector (and degraded muril): Hindi only.
    if "hi" not in languages:
        return None
    return "hi" if _wordlist_match(text) else None


def is_romanized_indic(
    text: str, languages: tuple[str, ...], *, detector: str = "word-list"
) -> bool:
    """Backward-compatible boolean wrapper over :func:`detect_target_language`."""
    return detect_target_language(text, languages, detector=detector) is not None
