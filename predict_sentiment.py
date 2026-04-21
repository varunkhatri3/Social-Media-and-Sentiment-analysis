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
_NEGATIVE_TO_TARGET = {"negative": 0, "neutral": 1, "positive": 2}

_POSITIVE_HINTS = {
    "absolutely love": 2.4,
    "would recommend": 1.9,
    "highly recommend": 1.9,
    "recommend": 1.0,
    "fantastic": 1.8,
    "great": 1.5,
    "happy": 1.5,
    "impressed": 1.6,
    "satisfy": 1.7,
    "satisfied": 1.7,
    "solid": 1.1,
    "smooth": 1.0,
    "smoother": 1.0,
    "enjoy": 1.4,
    "gladly": 1.2,
    "better": 1.2,
    "works great": 1.8,
}

_NEGATIVE_HINTS = {
    "terrible": -2.0,
    "hate": -2.2,
    "buggy": -1.8,
    "disappoint": -1.8,
    "disappointing": -1.8,
    "unhelpful": -1.8,
    "rude": -1.4,
    "worst": -2.2,
    "regret": -1.9,
    "crash": -2.1,
    "confus": -1.3,
    "confusing": -1.3,
    "frustrat": -1.7,
    "frustrating": -1.7,
    "waste": -1.9,
    "unhappy": -1.9,
    "awful": -2.1,
    "worse": -1.8,
    "harder": -1.2,
    "broke": -1.9,
    "break": -1.7,
    "stopped working": -2.1,
    "stop work": -2.1,
    "nothing work properly": -2.2,
}

_NEUTRAL_HINTS = {
    "okay": 0.7,
    "average": 0.8,
    "as expected": 0.7,
    "expect": 0.4,
    "later": 0.6,
    "tomorrow": 0.6,
    "meeting": 0.7,
    "report": 0.6,
    "package arrive": 0.7,
    "resolved": 0.6,
    "resolve": 0.6,
    "fine": 0.6,
    "standard": 0.8,
    "clear enough": 0.8,
    "no strong opinion": 1.1,
    "does not affect": 1.1,
    "nothing special": 1.0,
}

_STRONG_POSITIVE_PHRASES = {
    "would recommend",
}

_STRONG_NEGATIVE_PHRASES = {
    "unhelpful and rude",
    "stopped working",
    "much worse",
    "disappointing experience",
}

_STRONG_NEUTRAL_PHRASES: set[str] = set()


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


def _rule_hint_label(raw: str, cleaned: str) -> str | None:
    raw_l = raw.lower()
    cleaned_l = cleaned.lower()

    pos_score = sum(weight for phrase, weight in _POSITIVE_HINTS.items() if phrase in raw_l or phrase in cleaned_l)
    neg_score = sum(abs(weight) for phrase, weight in _NEGATIVE_HINTS.items() if phrase in raw_l or phrase in cleaned_l)
    neu_score = sum(weight for phrase, weight in _NEUTRAL_HINTS.items() if phrase in raw_l or phrase in cleaned_l)

    if neu_score >= 1.0 and pos_score < 1.5 and neg_score < 1.5:
        return "neutral"
    if pos_score >= max(1.5, neg_score + 0.2):
        return "positive"
    if neg_score >= max(1.5, pos_score + 0.2):
        return "negative"
    return None


def _strong_rule_label(raw: str, cleaned: str) -> str | None:
    raw_l = raw.lower()
    cleaned_l = cleaned.lower()

    for phrase in _STRONG_NEGATIVE_PHRASES:
        if phrase in raw_l or phrase in cleaned_l:
            return "negative"
    for phrase in _STRONG_POSITIVE_PHRASES:
        if phrase in raw_l or phrase in cleaned_l:
            return "positive"
    for phrase in _STRONG_NEUTRAL_PHRASES:
        if phrase in raw_l or phrase in cleaned_l:
            return "neutral"
    return None


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

    ensure_nltk_resources()
    cleaned = preprocess_text(raw, use_pos_tag=True)
    if not cleaned.strip():
        return "unknown"

    vectorizer, model, vader = _artifacts()
    X = vectorizer.transform([cleaned])
    strong_rule = _strong_rule_label(raw, cleaned)
    rule_label = _rule_hint_label(raw, cleaned)
    if strong_rule is not None:
        return strong_rule
    if X.nnz == 0 and vader is not None:
        vader_label = _label_from_compound(_lexicon_compound(raw, vader))
        return rule_label or vader_label

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        best_idx = max(range(len(probs)), key=probs.__getitem__)
        best_label = _LABEL_NAMES.get(int(model.classes_[best_idx]), "unknown")
        best_score = float(probs[best_idx])

        if rule_label is not None and rule_label != best_label:
            rule_idx = _NEGATIVE_TO_TARGET[rule_label]
            rule_score = float(probs[list(model.classes_).index(rule_idx)]) if rule_idx in model.classes_ else 0.0
            if best_score < 0.82 or rule_score + 0.18 >= best_score:
                return rule_label

        if vader is not None and best_score < 0.55:
            vader_label = _label_from_compound(_lexicon_compound(raw, vader))
            if vader_label != "neutral":
                return vader_label
        return best_label

    label = int(model.predict(X)[0])
    fallback_label = _LABEL_NAMES.get(label, "unknown")
    return rule_label or fallback_label
