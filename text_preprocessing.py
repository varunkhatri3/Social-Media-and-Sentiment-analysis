"""
Social-media text preprocessing: URL removal, English stopword removal, NLTK lemmatization.
Designed for input produced by `notebooks/sentiment_analysis.ipynb` (`data/sample_dataset.csv`: `text`, `sentiment`, `target` with 0/1/2 = neg/neutral/pos).
Preprocessing output is saved as `data/processed_dataset.csv` (see `notebooks/preprocessing.ipynb`).
Phase 3 TF-IDF uses column `text_clean` (same role as `cleaned_text` in generic Sentiment140 guides); see `notebooks/feature_extraction.ipynb`.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import nltk
from nltk.data import find
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

# URLs: http(s), www., and common bare-domain patterns after strip
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b(?:[\w-]+\.)+(?:com|net|org|io|co|gov|edu)\b/\S*",
    re.IGNORECASE,
)


def remove_urls(text: str) -> str:
    """Replace URLs with a space so token boundaries stay sane."""
    if not isinstance(text, str):
        text = str(text)
    return URL_RE.sub(" ", text)


def _treebank_to_wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return wn.ADJ
    if tag.startswith("V"):
        return wn.VERB
    if tag.startswith("N"):
        return wn.NOUN
    if tag.startswith("R"):
        return wn.ADV
    return wn.NOUN


@lru_cache(maxsize=1)
def _english_stopwords() -> frozenset[str]:
    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


def preprocess_text(
    text: str,
    *,
    use_pos_tag: bool = True,
    stop_words: Iterable[str] | None = None,
) -> str:
    """
    Full pipeline: remove URLs, lowercase, tokenize, drop stopwords, lemmatize.

    Parameters
    ----------
    use_pos_tag
        If True, use POS tags for WordNet lemmatization (slower, better quality).
        If False, lemmatize each token as a noun (faster for very large batches).
    """
    text = remove_urls(text)
    text = text.lower()
    tokens = word_tokenize(text)
    sw = _english_stopwords() if stop_words is None else frozenset(stop_words)
    # Keep tokens that contain at least one letter (keeps words; drops lone punctuation)
    tokens = [t for t in tokens if t not in sw and any(c.isalpha() for c in t)]
    if not tokens:
        return ""

    lemmatizer = _lemmatizer()

    if use_pos_tag:
        tagged = pos_tag(tokens)
        lemmas = [
            lemmatizer.lemmatize(word, _treebank_to_wordnet_pos(tag))
            for word, tag in tagged
        ]
    else:
        lemmas = [lemmatizer.lemmatize(w, wn.NOUN) for w in tokens]

    return " ".join(lemmas)


def ensure_nltk_resources() -> None:
    """Ensure required NLTK data exists locally before attempting downloads."""
    required = {
        "stopwords": ("corpora/stopwords",),
        "punkt": ("tokenizers/punkt",),
        "punkt_tab": ("tokenizers/punkt_tab",),
        "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
        "omw-1.4": ("corpora/omw-1.4", "corpora/omw-1.4.zip"),
        "averaged_perceptron_tagger": ("taggers/averaged_perceptron_tagger",),
        "averaged_perceptron_tagger_eng": ("taggers/averaged_perceptron_tagger_eng",),
    }
    for pkg, candidates in required.items():
        try:
            for resource_path in candidates:
                try:
                    find(resource_path)
                    break
                except LookupError:
                    continue
            else:
                nltk.download(pkg, quiet=True)
        except Exception:
            # Stay resilient in offline environments; downstream code will raise
            # a clearer error if a truly required resource is still missing.
            pass
