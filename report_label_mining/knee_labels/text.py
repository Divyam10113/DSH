"""Report text normalisation, sentence splitting and language detection.

Everything here is dependency-free so it runs in an offline Kaggle notebook.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_LIGATURES = {"ß": "ss", "œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae", "ø": "o", "Ø": "o"}
_LIGATURE_RE = re.compile("|".join(map(re.escape, _LIGATURES)))


def fold_text(text: str) -> str:
    """Lowercase, strip accents and collapse whitespace.

    Folding lets one regex cover e.g. "lésion"/"lesion" and "Außenmeniskus"/"aussenmeniskus".
    Newlines are kept because they often separate report sections.
    """
    if not isinstance(text, str):
        return ""
    text = _LIGATURE_RE.sub(lambda m: _LIGATURES[m.group(0)], text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*", "\n", text)
    return text.strip()


# Dotted abbreviations that would otherwise be split into sentences.
_ABBREVIATIONS = [
    (r"\bv\.\s?a\.", "verdacht auf"),
    (r"\bz\.\s?n\.", "zustand nach"),
    (r"\bst\.\s?n\.", "status nach"),
    (r"\bz\.\s?b\.", "zb"),
    (r"\be\.\s?g\.", "eg"),
    (r"\bi\.\s?e\.", "ie"),
    # Only abbreviations that practically never end a sentence ("5 mm." often does).
    (r"\b(ca|approx|vs|bzw|ggf|evtl|dr|lig|sog|resp)\.", r"\1"),
]
_ABBREVIATION_RES = [(re.compile(p, re.IGNORECASE), r) for p, r in _ABBREVIATIONS]

# Sentence ends: . ! ? ; or a newline. A period between digits (e.g. "3.5 mm") is not a boundary.
_SENT_SPLIT_RE = re.compile(r"(?<!\d)[.!?;](?!\d)\s*|\n+")


def split_sentences(text: str) -> list[str]:
    """Split report text into sentences (works on raw or `fold_text` output)."""
    for pattern, repl in _ABBREVIATION_RES:
        text = pattern.sub(repl, text)
    return [s.strip(" -•*\t") for s in _SENT_SPLIT_RE.split(text) if s and s.strip(" -•*\t")]


# Small stopword profiles for the languages we expect in the reports.
_STOPWORDS = {
    "en": "the of and with no is are there in to a or without normal tear intact seen",
    "de": "der die das und mit kein keine ist im nicht ohne eine des zur regelrecht unauffallig",
    "es": "el la los las de del y con sin no se en es una por ligamento rotura",
    "fr": "le la les de des du et avec sans pas une est au aux ligament dechirure",
    "it": "il lo la gli le di del della e con senza non si una legamento rottura",
    "pt": "o a os as de do da e com sem nao em uma ligamento rotura",
    "nl": "de het een en met geen niet van is in zonder band",
}
_STOPWORD_SETS = {lang: set(words.split()) for lang, words in _STOPWORDS.items()}
_WORD_RE = re.compile(r"[a-z]+")


def _heuristic_language(folded: str) -> str:
    counts = Counter(_WORD_RE.findall(folded))
    if not counts:
        return "unk"
    scores = {lang: sum(counts[w] for w in words) for lang, words in _STOPWORD_SETS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unk"


def detect_language(text: str) -> str:
    """ISO-639-1 code of the report language, or "unk".

    Uses `langdetect` when it is installed (dev notebooks), otherwise a stopword heuristic that is
    good enough for the European languages this dataset is likely to contain.
    """
    folded = fold_text(text)
    if not folded:
        return "unk"
    try:
        from langdetect import DetectorFactory, detect  # type: ignore

        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return _heuristic_language(folded)
