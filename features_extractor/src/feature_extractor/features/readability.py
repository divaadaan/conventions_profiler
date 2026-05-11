"""Readability formula features.

Operates directly on essay.text — no spaCy or LanguageTool dependency, no
shared state, just textstat formulas. Three features:

    read_flesch_reading_ease       Flesch reading ease score (higher = easier)
    read_flesch_kincaid_grade      US grade level estimate
    read_dale_chall_difficult_word_density   PDW: density of words outside the
                                              Dale-Chall list of "easy" words

References:
    Flesch (1948), Dale & Chall (1948). See docs.
"""

import textstat

from feature_extractor.models import Essay


def compute_readability_features(essay: Essay) -> dict[str, float]:
    text = essay.text or ""
    if not text.strip():
        return _empty_features()

    # textstat returns floats; some return 0.0 on empty/short text.
    fre = float(textstat.flesch_reading_ease(text))
    fkg = float(textstat.flesch_kincaid_grade(text))

    difficult = textstat.difficult_words(text)
    total_words = textstat.lexicon_count(text, removepunct=True)
    pdw = (difficult / total_words) if total_words > 0 else 0.0

    return {
        "read_flesch_reading_ease": fre,
        "read_flesch_kincaid_grade": fkg,
        "read_dale_chall_difficult_word_density": float(pdw),
    }


def _empty_features() -> dict[str, float]:
    return {
        "read_flesch_reading_ease": 0.0,
        "read_flesch_kincaid_grade": 0.0,
        "read_dale_chall_difficult_word_density": 0.0,
    }
