"""
End-to-end sentiment prediction: same preprocessing as training, then TF-IDF + saved classifier.

Labels match the dataset: 0 = negative, 1 = neutral, 2 = positive.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math
import re

import joblib

from text_preprocessing import ensure_nltk_resources, preprocess_text

_PROJECT_ROOT = Path(__file__).resolve().parent
_VECTORIZER_PATH = _PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
_MODEL_PATH = _PROJECT_ROOT / "models" / "best_model.pkl"
_VADER_LEXICON_PATH = _PROJECT_ROOT / "models" / "vader_lexicon.txt"

_LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


@lru_cache(maxsize=1)
def _artifacts():
    ensure_nltk_resources()
    if not _VECTORIZER_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {_VECTORIZER_PATH}. Run notebooks/feature_extraction.ipynb first."
        )
    if not _MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {_MODEL_PATH}. Run notebooks/model_training.ipynb first."
        )
    vectorizer = joblib.load(_VECTORIZER_PATH)
    model = joblib.load(_MODEL_PATH)
    vader = None
    if _VADER_LEXICON_PATH.is_file():
        vader = _load_vader_lexicon(_VADER_LEXICON_PATH)
    return vectorizer, model, vader


def _label_from_compound(compound: float) -> str:
    if compound >= 0.3:
        return "positive"
    if compound <= -0.3:
        return "negative"
    return "neutral"


def _load_vader_lexicon(path: Path) -> dict[str, float]:
    lexicon: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            lexicon[parts[0]] = float(parts[1])
    return lexicon


def _lexicon_compound(text: str, lexicon: dict[str, float]) -> float:
    tokens = re.findall(r"[a-z][a-z'_-]*", text.lower())
    if not tokens:
        return 0.0
    score = sum(lexicon.get(token, 0.0) for token in tokens)
    if score == 0.0:
        return 0.0
    return score / math.sqrt(score * score + 15)


def predict_sentiment(text: str) -> str:
    """
    Predict sentiment for raw social text.

    Applies URL removal, stopword removal, and NLTK lemmatization (POS-aware),
    then the fitted TF-IDF vectorizer and saved multiclass model.

    Returns
    -------
    \"negative\" | \"neutral\" | \"positive\" | \"unknown\"
    """
    raw = str(text).strip()
    if not raw:
        return "unknown"

    cleaned = preprocess_text(raw, use_pos_tag=True)
    if not cleaned.strip():
        return "unknown"

    vectorizer, model, vader = _artifacts()
    X = vectorizer.transform([cleaned])
    if X.nnz == 0 and vader is not None:
        return _label_from_compound(_lexicon_compound(raw, vader))

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        best_idx = max(range(len(probs)), key=probs.__getitem__)
        best_label = _LABEL_NAMES.get(int(model.classes_[best_idx]), "unknown")
        best_score = float(probs[best_idx])

        if vader is not None and best_score < 0.55:
            vader_label = _label_from_compound(_lexicon_compound(raw, vader))
            if vader_label != "neutral":
                return vader_label
        return best_label

    label = int(model.predict(X)[0])
    return _LABEL_NAMES.get(label, "unknown")
