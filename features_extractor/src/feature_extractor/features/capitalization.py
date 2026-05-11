"""Capitalization features decomposed by curriculum concept.

Three error rates, each anchored to the relevant denominator:
    cap_sentence_initial_rate   sentence-initial lowercase / num_sentences
    cap_proper_noun_rate        lowercase proper nouns / num_proper_nouns
    cap_intrusion_rate          mid-sentence unexpected capitals / num_sentences
"""

from feature_extractor.models import Essay

# Strict set: NER types that are unambiguously proper-noun-like. Used as the
# denominator for cap_proper_noun_rate.
_PROPER_NOUN_ENT_TYPES = {
    "PERSON", "GPE", "LOC", "ORG", "NORP",
    "FAC", "EVENT", "WORK_OF_ART", "LANGUAGE", "PRODUCT",
}

# DATE/TIME entities are conflated by NER between proper-noun-like terms
# (weekdays, months, holidays) and common time words ("noon", "today",
# "yesterday"). Use a positive whitelist to distinguish — only words in this
# set are exempt from intrusion when NER tags them DATE/TIME.
_CAPITALIZABLE_DATE_TIME_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "christmas", "easter", "halloween", "thanksgiving", "hanukkah",
    "ramadan", "diwali", "kwanzaa", "passover",
}


def _is_proper_noun(token: dict) -> bool:
    """Detect proper nouns via NER only.

    POS=PROPN is intentionally NOT used: spaCy's POS tagger uses casing as a
    strong feature, so capitalized common nouns ('Park', 'Noon') frequently
    get tagged PROPN purely on casing. NER predicts entity status from
    broader context and is more robust to both miscapitalized inputs and
    common-noun-as-PROPN false positives.
    """
    return token.get("ent_type") in _PROPER_NOUN_ENT_TYPES


def _is_intrusion_exempt(token: dict) -> bool:
    """A token is exempt from intrusion detection if either:
      - NER tags it as a proper-noun entity type, OR
      - NER tags it DATE/TIME and its lower form is in the capitalizable
        whitelist (Saturday, January, Christmas, etc.).

    The DATE/TIME whitelist guards against NER over-exempting common time
    words like 'noon', 'today', 'yesterday', which should stay lowercase.
    """
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

        # Sentence-initial check
        if first_alpha_i is not None:
            first = tokens[first_alpha_i]
            if first["text"] and not first["text"][0].isupper():
                sent_initial_errors += 1

        # Per-token proper-noun and intrusion checks
        for i, tok in enumerate(tokens):
            if not tok["is_alpha"]:
                continue

            text = tok["text"]
            if not text:
                continue

            is_sent_initial = (i == first_alpha_i)
            starts_upper = text[0].isupper()

            # Proper noun rate excludes sentence-initial position to avoid
            # double-counting against cap_sentence_initial_rate.
            if _is_proper_noun(tok) and not is_sent_initial:
                proper_noun_total += 1
                if not starts_upper:
                    proper_noun_errors += 1

            # Intrusion: mid-sentence unexpected capital. Excludes:
            #   - sentence-initial position (caught by cap_sentence_initial_rate)
            #   - NER-confirmed entities (PERSON, GPE, DATE, etc.)
            #   - the pronoun "I" (always capitalized; counted by LT I_LOWERCASE)
            #   - acronyms (NASA, USA — legitimate all-caps)
            if (
                not is_sent_initial
                and starts_upper
                and not _is_intrusion_exempt(tok)
                and text != "I"
                and not _is_acronym(text)
            ):
                intrusion_count += 1

    num_sentences = len(sentences)

    return {
        "cap_sentence_initial_rate": sent_initial_errors / num_sentences,
        "cap_proper_noun_rate": (
            proper_noun_errors / proper_noun_total if proper_noun_total > 0 else 0.0
        ),
        "cap_intrusion_rate": intrusion_count / num_sentences,
    }


def _empty_features() -> dict[str, float]:
    return {
        "cap_sentence_initial_rate": 0.0,
        "cap_proper_noun_rate": 0.0,
        "cap_intrusion_rate": 0.0,
    }
