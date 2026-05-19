"""Spelling features distinguishing meaning-disrupting from cosmetic errors."""

from spellchecker import SpellChecker

from feature_extractor.models import Essay

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
FUNCTION_POS = {"DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "PRON"}

MIN_WORD_LENGTH = 2

_spell = SpellChecker()


def _extract_words_with_pos(essay: Essay) -> list[dict]:
    spacy_data = essay.pass_results.get("spacy", {})
    sentences = spacy_data.get("sentences", [])

    words = []
    for sent in sentences:
        for tok in sent.get("tokens", []):
            if tok["is_alpha"] and len(tok["lower"]) >= MIN_WORD_LENGTH:
                words.append({
                    "word": tok["lower"],
                    "original": tok["text"],
                    "pos": tok["pos"],
                    "lemma": tok.get("lemma", tok["lower"]),
                })

    return words


def compute_spelling_features(essay: Essay) -> dict[str, float]:
    words_with_pos = _extract_words_with_pos(essay)

    if not words_with_pos:
        return _empty_features()

    features: dict[str, float] = {}

    word_list = [w["word"] for w in words_with_pos]
    unknown = _spell.unknown(word_list)

    misspellings = []
    for w in words_with_pos:
        if w["word"] in unknown:
            correction = _spell.correction(w["word"])
            misspellings.append({
                "word": w["word"],
                "lemma": w["lemma"],
                "pos": w["pos"],
                "correction": correction,
                "has_close_match": correction is not None and correction != w["word"],
            })

    total_words = len(words_with_pos)
    misspelled_count = len(misspellings)

    features["spell_misspelled_count"] = float(misspelled_count)
    features["spell_misspelled_ratio"] = (
        misspelled_count / total_words if total_words > 0 else 0.0
    )

    content_misspellings = sum(1 for m in misspellings if m["pos"] in CONTENT_POS)
    function_misspellings = sum(1 for m in misspellings if m["pos"] in FUNCTION_POS)

    features["spell_content_word_errors"] = float(content_misspellings)
    features["spell_function_word_errors"] = float(function_misspellings)

    features["spell_content_error_ratio"] = (
        content_misspellings / misspelled_count if misspelled_count > 0 else 0.0
    )

    close_match_count = sum(1 for m in misspellings if m["has_close_match"])
    features["spell_close_match_ratio"] = (
        close_match_count / misspelled_count if misspelled_count > 0 else 0.0
    )

    far_misspellings = misspelled_count - close_match_count
    features["spell_far_misspelling_count"] = float(far_misspellings)
    features["spell_far_misspelling_ratio"] = (
        far_misspellings / total_words if total_words > 0 else 0.0
    )

    # Lemma dedup: 'becuase' and 'becuases' count as one error type.
    unique_misspelled_lemmas = {m["lemma"] for m in misspellings}
    unique_misspelled_count = len(unique_misspelled_lemmas)
    features["spell_unique_misspelled_lemma_count"] = float(unique_misspelled_count)
    features["spell_unique_misspelled_lemma_ratio"] = (
        unique_misspelled_count / total_words if total_words > 0 else 0.0
    )

    unique_content_lemmas = {m["lemma"] for m in misspellings if m["pos"] in CONTENT_POS}
    unique_function_lemmas = {m["lemma"] for m in misspellings if m["pos"] in FUNCTION_POS}
    features["spell_unique_content_lemma_count"] = float(len(unique_content_lemmas))
    features["spell_unique_content_lemma_ratio"] = (
        len(unique_content_lemmas) / total_words if total_words > 0 else 0.0
    )
    features["spell_unique_function_lemma_count"] = float(len(unique_function_lemmas))
    features["spell_unique_function_lemma_ratio"] = (
        len(unique_function_lemmas) / total_words if total_words > 0 else 0.0
    )

    # Positive framing: rubric scores 30/40 expect visible errors, so the clean
    # signal aggregate is more discriminating than the error rate.
    features["spell_unique_correct_lemma_rate"] = (
        1.0 - features["spell_unique_misspelled_lemma_ratio"]
    )

    # >1 means the same misspelled lemma repeats; ==1 means each error is unique.
    raw_ratio = features["spell_misspelled_ratio"]
    unique_ratio = features["spell_unique_misspelled_lemma_ratio"]
    features["spell_repetition_factor"] = (
        raw_ratio / unique_ratio if unique_ratio > 0 else 0.0
    )

    return features


def _empty_features() -> dict[str, float]:
    keys = [
        "spell_misspelled_count", "spell_misspelled_ratio",
        "spell_content_word_errors", "spell_function_word_errors",
        "spell_content_error_ratio",
        "spell_close_match_ratio",
        "spell_far_misspelling_count", "spell_far_misspelling_ratio",
        "spell_unique_misspelled_lemma_count",
        "spell_unique_misspelled_lemma_ratio",
        "spell_unique_content_lemma_count",
        "spell_unique_content_lemma_ratio",
        "spell_unique_function_lemma_count",
        "spell_unique_function_lemma_ratio",
        "spell_unique_correct_lemma_rate",
        "spell_repetition_factor",
    ]
    empty = {k: 0.0 for k in keys}
    empty["spell_unique_correct_lemma_rate"] = 1.0
    return empty
