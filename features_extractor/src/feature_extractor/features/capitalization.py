"""Capitalization features decomposed by curriculum concept.

Three error rates, each anchored to the relevant denominator:
    cap_sentence_initial_rate   sentence-initial lowercase / num_sentences
    cap_proper_noun_rate        lowercase proper nouns / num_proper_nouns
    cap_intrusion_rate          mid-sentence unexpected capitals / num_sentences
"""

from feature_extractor.models import Essay

_PROPER_NOUN_ENT_TYPES = {
    "PERSON", "GPE", "LOC", "ORG", "NORP",
    "FAC", "EVENT", "WORK_OF_ART", "LANGUAGE", "PRODUCT",
}

# NER tags DATE/TIME on both proper nouns (weekdays, months, holidays) and
# common time words ("noon", "today"). Whitelist gates the exemption to the
# former.
_CAPITALIZABLE_DATE_TIME_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "christmas", "easter", "halloween", "thanksgiving", "hanukkah",
    "ramadan", "diwali", "kwanzaa", "passover",
}


def _is_proper_noun(token: dict) -> bool:
    # NER, not POS=PROPN: spaCy's POS tagger keys on casing, so capitalized
    # common nouns ('Park', 'Noon') get tagged PROPN on casing alone.
    return token.get("ent_type") in _PROPER_NOUN_ENT_TYPES


def _is_intrusion_exempt(token: dict) -> bool:
    ent = token.get("ent_type")
    if ent in _PROPER_NOUN_ENT_TYPES:
        return True
    if ent in {"DATE", "TIME"} and token.get("lower") in _CAPITALIZABLE_DATE_TIME_WORDS:
        return True
    return False


def _is_acronym(text: str) -> bool:
    return len(text) > 1 and text.isupper() and text.isalpha()


def _first_alpha_index(tokens: list[dict]) -> int | None:
    for i, t in enumerate(tokens):
        if t["is_alpha"]:
            return i
    return None


def compute_capitalization_features(essay: Essay) -> dict[str, float]:
    spacy_data = essay.pass_results.get("spacy", {})
    sentences = spacy_data.get("sentences", [])

    if not sentences:
        return _empty_features()

    sent_initial_errors = 0
    proper_noun_total = 0
    proper_noun_errors = 0
    intrusion_count = 0

    for sent in sentences:
        tokens = sent.get("tokens", [])
        first_alpha_i = _first_alpha_index(tokens)

        if first_alpha_i is not None:
            first = tokens[first_alpha_i]
            if first["text"] and not first["text"][0].isupper():
                sent_initial_errors += 1

        for i, tok in enumerate(tokens):
            if not tok["is_alpha"]:
                continue

            text = tok["text"]
            if not text:
                continue

            is_sent_initial = (i == first_alpha_i)
            starts_upper = text[0].isupper()

            # Skip sentence-initial here so it's not double-counted against
            # cap_sentence_initial_rate.
            if _is_proper_noun(tok) and not is_sent_initial:
                proper_noun_total += 1
                if not starts_upper:
                    proper_noun_errors += 1

            # "I" is counted by LT's I_LOWERCASE; acronyms are legitimate.
            if (
                not is_sent_initial
                and starts_upper
                and not _is_intrusion_exempt(tok)
                and text != "I"
                and not _is_acronym(text)
            ):
                intrusion_count += 1

    num_sentences = len(sentences)

    sent_initial_rate = sent_initial_errors / num_sentences
    proper_noun_rate = (
        proper_noun_errors / proper_noun_total if proper_noun_total > 0 else 0.0
    )
    intrusion_rate = intrusion_count / num_sentences

    return {
        "cap_sentence_initial_rate": sent_initial_rate,
        "cap_proper_noun_rate": proper_noun_rate,
        "cap_intrusion_rate": intrusion_rate,
        "cap_sentence_initial_correct_rate": 1.0 - sent_initial_rate,
        "cap_proper_noun_correct_rate": 1.0 - proper_noun_rate,
        # intrusion_rate can exceed 1 (multiple intrusions per sentence), clamp before flipping.
        "cap_intrusion_correct_rate": max(0.0, 1.0 - min(intrusion_rate, 1.0)),
    }


def _empty_features() -> dict[str, float]:
    return {
        "cap_sentence_initial_rate": 0.0,
        "cap_proper_noun_rate": 0.0,
        "cap_intrusion_rate": 0.0,
        "cap_sentence_initial_correct_rate": 1.0,
        "cap_proper_noun_correct_rate": 1.0,
        "cap_intrusion_correct_rate": 1.0,
    }
