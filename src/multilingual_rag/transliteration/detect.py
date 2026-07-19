"""Detect romanized Hindi — decides *whether* to transliterate a Latin-script query.

Score-based routing proved unreliable at scale (the raw romanized search finds enough
high-cosine noise to look confident). A cheap linguistic check is far more reliable: romanized
Hindi is saturated with function words that essentially never appear in English (`kya`, `hai`,
`kaun`, `kahan`, `nahi`, …). We deliberately exclude tokens that collide with English (`is`,
`to`, `me`, `the`, `par`, `ka`, `ki`) to keep **precision** high — a false positive would
mis-transliterate a real English query, so we only fire on distinctly-Hindi evidence.
"""

from __future__ import annotations

import re

from multilingual_rag.transliteration.script import is_latin_script

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


def is_romanized_indic(text: str, languages: tuple[str, ...]) -> bool:
    """True when ``text`` looks like romanized Hindi that should be transliterated.

    Only Hindi is implemented; kn/te would need their own marker sets. Requires Latin script
    (a native-Devanagari query is already fine) plus at least one distinctly-Hindi token.
    """
    if "hi" not in languages or not is_latin_script(text):
        return False
    return any(word in _HINDI_MARKERS for word in _WORD_RE.findall(text.lower()))
