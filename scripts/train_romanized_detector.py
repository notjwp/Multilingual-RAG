"""Train the opt-in MuRIL romanized-Hindi detector (frozen MuRIL features + LogisticRegression).

Positives: romanized XQuAD-hi questions (how people type). Negatives: XQuAD en + es questions
(English *and* Spanish, so the head learns "Latin-script ≠ Hindi", not merely "not English").
MuRIL is used frozen — only the tiny LR head is trained and committed
(`data/models/romanized_hi_detector.joblib`). The threshold is tuned for ~0 false positives, to
match the word-list detector's precision-first stance.

This is a one-off offline step; the runtime detector (`transliteration/detect.py`, opt-in via
`TRANSLITERATION_DETECTOR=muril`) loads the produced artifact.

Usage:
    python scripts/train_romanized_detector.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_romanized import romanize  # noqa: E402  (sibling script)

from multilingual_rag.transliteration.detect import is_romanized_indic  # noqa: E402
from multilingual_rag.transliteration.muril import MurilFeatureExtractor  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "eval" / "xquad"
ARTIFACT = Path(__file__).resolve().parents[1] / "data" / "models" / "romanized_hi_detector.joblib"
SEED = 42


def _questions(lang: str) -> list[str]:
    path = DATA_DIR / f"queries_{lang}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["question"] for line in lines if line.strip()]


def main() -> None:
    # Labeled data: romanized Hindi (1) vs Latin-script non-Hindi (0).
    positives = [romanize(q) for q in _questions("hi")]
    negatives = _questions("en") + _questions("es")
    texts = positives + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives))
    print(f"positives (romanized-hi): {len(positives)} | negatives (en+es): {len(negatives)}")

    print("embedding with MuRIL (frozen)...")
    features = MurilFeatureExtractor().embed(texts)

    x_train, x_test, y_train, y_test, _, text_test = train_test_split(
        features, labels, texts, test_size=0.2, stratify=labels, random_state=SEED
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train, y_train)

    # Tune the threshold on held-out for ~0 false positives (precision-first, like the word list):
    # any threshold above the highest-scoring negative yields 0 FP.
    proba = clf.predict_proba(x_test)[:, 1]
    neg_max = float(proba[y_test == 0].max())
    threshold = float(np.nextafter(neg_max, 1.0))
    predicted = proba >= threshold
    recall = float((predicted[y_test == 1]).mean())
    fp_rate = float((predicted[y_test == 0]).mean())

    # Word-list baseline on the SAME held-out texts.
    wl = np.array([is_romanized_indic(t, ("hi",), detector="word-list") for t in text_test])
    wl_recall = float(wl[y_test == 1].mean())
    wl_fp = float(wl[y_test == 0].mean())

    print(f"\n{'detector':<12}{'recall':>10}{'FP':>10}{'threshold':>12}")
    print("-" * 44)
    print(f"{'MuRIL+LR':<12}{recall:>10.3f}{fp_rate:>10.3f}{threshold:>12.4f}")
    print(f"{'word-list':<12}{wl_recall:>10.3f}{wl_fp:>10.3f}{'—':>12}")

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "threshold": threshold}, ARTIFACT)
    print(f"\nsaved head → {ARTIFACT.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
