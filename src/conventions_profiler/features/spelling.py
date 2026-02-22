"""
Spelling features distinguishing meaning-disrupting from cosmetic errors.

Uses pyspellchecker for dictionary-based spell checking and edit distance
computation. Leverages POS tags from the spaCy pass to classify misspellings
by word type (content words vs. function words).

Key distinction: content-word misspellings (nouns, verbs, adjectives, adverbs)
are more damaging to comprehension than function-word misspellings because
content words carry the semantic payload of the sentence.
"""

import re

from spellchecker import SpellChecker

from conventions_profiler.models import Essay

# POS tags for content words (carry semantic meaning)
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

# POS tags for function words (grammatical role, less semantic impact)
FUNCTION_POS = {"DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "PRON"}

# Minimum word length to consider for spell checking
MIN_WORD_LENGTH = 2

# Initialize spell checker once at module level
_spell = SpellChecker()


def _extract_words_with_pos(essay: Essay) -> list[dict]:
    """Extract words with their POS tags from spaCy pass results.

    Returns a list of dicts with 'word' (lowercased), 'pos', and 'original'.
    Filters to alphabetic tokens of sufficient length.
    """
    spacy_data = essay.pass_results.get("spacy", {})
    sentences = spacy_data.get("sentences", [])

    # We need the raw text tokens — reconstruct from essay text using
    # POS sequences from spaCy as a guide
    words = []
    # Simple tokenization aligned with spaCy's output
    raw_tokens = re.findall(r"[a-zA-Z]+", essay.text)

    # Get all POS tags in order
    all_pos = []
    for sent in sentences:
        all_pos.extend(sent.get("pos_sequence", []))

    # Align raw alphabetic tokens with POS tags
    # This is approximate — spaCy tokenizes differently than regex
    pos_idx = 0
    for token in raw_tokens:
        if len(token) < MIN_WORD_LENGTH:
            pos_idx += 1
            continue

        pos = all_pos[pos_idx] if pos_idx < len(all_pos) else "X"
        words.append({
            "word": token.lower(),
            "original": token,
            "pos": pos,
        })
        pos_idx += 1

    return words


def compute_spelling_features(essay: Essay) -> dict[str, float]:
    """Extract spelling features using pyspellchecker and spaCy POS tags."""
    words_with_pos = _extract_words_with_pos(essay)

    if not words_with_pos:
        return _empty_features()

    features: dict[str, float] = {}

    # Check all words against dictionary
    word_list = [w["word"] for w in words_with_pos]
    unknown = _spell.unknown(word_list)

    # Build misspelling details
    misspellings = []
    for w in words_with_pos:
        if w["word"] in unknown:
            correction = _spell.correction(w["word"])
            edit_dist = _spell.edit_distance_1(w["word"])
            misspellings.append({
                "word": w["word"],
                "pos": w["pos"],
                "correction": correction,
                "has_close_match": len(edit_dist) > 0,
            })

    total_words = len(words_with_pos)
    misspelled_count = len(misspellings)

    # --- Basic spelling counts ---
    features["spell_misspelled_count"] = float(misspelled_count)
    features["spell_misspelled_ratio"] = (
        misspelled_count / total_words if total_words > 0 else 0.0
    )

    # --- Content word vs function word misspellings ---
    content_misspellings = sum(1 for m in misspellings if m["pos"] in CONTENT_POS)
    function_misspellings = sum(1 for m in misspellings if m["pos"] in FUNCTION_POS)

    features["spell_content_word_errors"] = float(content_misspellings)
    features["spell_function_word_errors"] = float(function_misspellings)

    # Ratio of content word misspellings to total misspellings
    features["spell_content_error_ratio"] = (
        content_misspellings / misspelled_count if misspelled_count > 0 else 0.0
    )

    # --- Edit distance features ---
    # Words with close matches (edit distance 1) are more recoverable by readers
    close_match_count = sum(1 for m in misspellings if m["has_close_match"])
    features["spell_close_match_ratio"] = (
        close_match_count / misspelled_count if misspelled_count > 0 else 0.0
    )

    # Words without close matches are harder to recover — more damaging
    far_misspellings = misspelled_count - close_match_count
    features["spell_far_misspelling_count"] = float(far_misspellings)
    features["spell_far_misspelling_ratio"] = (
        far_misspellings / total_words if total_words > 0 else 0.0
    )

    return features


def _empty_features() -> dict[str, float]:
    keys = [
        "spell_misspelled_count", "spell_misspelled_ratio",
        "spell_content_word_errors", "spell_function_word_errors",
        "spell_content_error_ratio",
        "spell_close_match_ratio",
        "spell_far_misspelling_count", "spell_far_misspelling_ratio",
    ]
    return {k: 0.0 for k in keys}
