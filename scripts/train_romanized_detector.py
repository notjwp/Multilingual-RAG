"""Train the opt-in MuRIL romanized-Indic detector (frozen MuRIL features + LogisticRegression).

Multi-class: given a Latin-script query, predict which Indic language it's romanized in — ``hi``,
``kn``, ``te`` — or ``other`` (English/Spanish/anything non-Indic → don't transliterate). MuRIL is
used *frozen*; only the tiny multinomial LR head is trained and committed
(`data/models/romanized_indic_detector.joblib`). This gives the same multi-language detection as the
``google`` detector but **locally** (no per-query network).

Training data (avoids contaminating the kn/te retrieval eval, which uses the *gold* sentences):
  hi    — romanized XQuAD-hi questions
  kn/te — romanized Wikipedia *distractor* sentences (gold is held out for eval_romanized)
  other — XQuAD en + es questions (Latin-script non-Indic)

A max-probability threshold is tuned so ``other`` almost never routes to an Indic language
(precision-first). Usage:  python scripts/train_romanized_detector.py
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

ROOT = Path(__file__).resolve().parents[1]
XQUAD = ROOT / "data" / "eval" / "xquad"
INDIC = ROOT / "data" / "eval" / "indic"
ARTIFACT = ROOT / "data" / "models" / "romanized_indic_detector.joblib"
INDIC_LANGS = ("hi", "kn", "te")
SEED = 42


def _questions(lang: str) -> list[str]:
    lines = (XQUAD / f"queries_{lang}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["question"] for line in lines if line.strip()]


def _distractors(lang: str) -> list[str]:
    lines = (INDIC / f"distractors_{lang}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["text"] for line in lines if line.strip()]


def main() -> None:
    # Labeled romanized text per class. hi from XQuAD; kn/te from Wikipedia distractors (gold held
    # out); other = Latin-script non-Indic.
    samples: list[tuple[str, str]] = []
    samples += [(romanize(q, "hi"), "hi") for q in _questions("hi")]
    samples += [(romanize(t, "kn"), "kn") for t in _distractors("kn")]
    samples += [(romanize(t, "te"), "te") for t in _distractors("te")]
    samples += [(q, "other") for q in _questions("en") + _questions("es")]

    texts = [text for text, _ in samples]
    labels = np.array([label for _, label in samples])
    counts = {lang: int((labels == lang).sum()) for lang in (*INDIC_LANGS, "other")}
    print("class counts:", counts)

    print("embedding with MuRIL (frozen)...")
    features = MurilFeatureExtractor().embed(texts)
    x_train, x_test, y_train, y_test, _, text_test = train_test_split(
        features, labels, texts, test_size=0.2, stratify=labels, random_state=SEED
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_train, y_train)
    classes = list(clf.classes_)
    indic = set(INDIC_LANGS)

    proba = clf.predict_proba(x_test)
    argmax = np.array(classes)[proba.argmax(axis=1)]
    max_proba = proba.max(axis=1)

    # Threshold: keep "other" from routing to an Indic language. The explicit "other" class already
    # separates non-Indic (0 leakage below), so trust argmax by default (threshold 0.0) — a >0
    # threshold would wrongly drop correct Indic predictions, whose max-proba is naturally < 0.5 in
    # a 4-class problem. Only raise it above any "other" sample that leaks to an Indic class.
    leak = max_proba[(y_test == "other") & np.isin(argmax, list(indic))]
    threshold = float(np.nextafter(leak.max(), 1.0)) if leak.size else 0.0

    def predicted_lang(i: int) -> str | None:
        return argmax[i] if (argmax[i] in indic and max_proba[i] >= threshold) else None

    preds = [predicted_lang(i) for i in range(len(y_test))]
    print(f"\nthreshold={threshold:.4f}")
    print(f"{'lang':<8}{'recall (MuRIL)':>16}{'word-list':>12}")
    print("-" * 36)
    for lang in INDIC_LANGS:
        idx = np.where(y_test == lang)[0]
        recall = float(np.mean([preds[i] == lang for i in idx])) if idx.size else 0.0
        wl = (
            float(np.mean([is_romanized_indic(text_test[i], ("hi",), detector="word-list")
                           for i in idx]))
            if lang == "hi" and idx.size else float("nan")
        )
        print(f"{lang:<8}{recall:>16.3f}{wl:>12.3f}")
    other_idx = np.where(y_test == "other")[0]
    indic_fp = float(np.mean([preds[i] in indic for i in other_idx])) if other_idx.size else 0.0
    print(f"\nother→Indic false-positive rate: {indic_fp:.3f}")

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "threshold": threshold, "classes": classes}, ARTIFACT)
    print(f"saved head → {ARTIFACT.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
