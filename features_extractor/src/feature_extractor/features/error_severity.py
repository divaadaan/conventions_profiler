"""Error severity features implementing the global/local error framework.

Severity tiers and weights are defaults validated against the training corpus.
"""

from feature_extractor.models import Essay

GLOBAL_CATEGORIES = {
    "CONFUSED_WORDS",
}

GLOBAL_RULES = {
    "THEIR_IS", "THERE_THEIR", "YOUR_YOU_2", "YOUR_YOU_RE", "YOUR",
    "TO_TOO", "IT_IS", "IT_IS_2", "ITS_TO_IT_S",
    "AFFECT_EFFECT", "LOOSE_LOSE", "ADVICE_ADVISE",
    "LESS_MORE_THEN", "RATHER_THEN", "THOUGH_THOUGHT",
    "COMMA_COMPOUND_SENTENCE", "COMMA_COMPOUND_SENTENCE_2",
    "AGREEMENT_SENT_START", "SINGULAR_AGREEMENT_SENT_START",
    "HE_VERB_AGR", "PEOPLE_VBZ",
    "MISSING_TO_BEFORE_A_VERB", "PRP_PAST_PART",
    "MD_BASEFORM", "MD_BE_NON_VBP",
    "THE_CC", "THE_ARE",
    "DOUBLE_NEGATIVE",
}

LOCAL_CATEGORIES = {
    "TYPOS",
    "CASING",
    "COMPOUNDING",
    "REDUNDANCY",
    "STYLE",
    "TYPOGRAPHY",
    "REPETITIONS_STYLE",
}

LOCAL_RULES = {
    "UPPERCASE_SENTENCE_START",
    "EN_UNPAIRED_BRACKETS",
    "EN_UNPAIRED_QUOTES",
    "WHITESPACE_RULE",
    "SENTENCE_WHITESPACE",
    "SPACE_BEFORE_PARENTHESIS",
    "COMMA_PARENTHESIS_WHITESPACE",
    "DOUBLE_PUNCTUATION",
    "MORFOLOGIK_RULE_EN_US",
    "EN_CONTRACTION_SPELLING",
}

SEVERITY_WEIGHTS = {
    "global": 3.0,
    "medium": 2.0,
    "local": 1.0,
}


def _classify_severity(match: dict) -> str:
    rule_id = match.get("rule_id", "")
    category = match.get("category", "")

    if rule_id in GLOBAL_RULES:
        return "global"
    if rule_id in LOCAL_RULES:
        return "local"

    if category in GLOBAL_CATEGORIES:
        return "global"
    if category in LOCAL_CATEGORIES:
        return "local"

    return "medium"


def compute_error_severity_features(essay: Essay) -> dict[str, float]:
    lt_data = essay.pass_results.get("languagetool", {})
    matches = lt_data.get("matches", [])
    word_count = essay.word_count

    features: dict[str, float] = {"err_total_count": float(len(matches)),
                                  "err_total_density": len(matches) / word_count if word_count > 0 else 0.0}

    severity_counts = {"global": 0, "medium": 0, "local": 0}
    for match in matches:
        tier = _classify_severity(match)
        severity_counts[tier] += 1

    features["err_global_count"] = float(severity_counts["global"])
    features["err_medium_count"] = float(severity_counts["medium"])
    features["err_local_count"] = float(severity_counts["local"])

    features["err_global_density"] = (
        severity_counts["global"] / word_count if word_count > 0 else 0.0
    )
    features["err_medium_density"] = (
        severity_counts["medium"] / word_count if word_count > 0 else 0.0
    )
    features["err_local_density"] = (
        severity_counts["local"] / word_count if word_count > 0 else 0.0
    )

    weighted_sum = sum(
        severity_counts[tier] * SEVERITY_WEIGHTS[tier]
        for tier in severity_counts
    )
    features["err_weighted_density"] = weighted_sum / word_count if word_count > 0 else 0.0

    total_errors = len(matches)
    features["err_global_ratio"] = (
        severity_counts["global"] / total_errors if total_errors > 0 else 0.0
    )

    categories = lt_data.get("categories", {})
    for cat_name in ["GRAMMAR", "TYPOS", "PUNCTUATION", "CASING",
                     "CONFUSED_WORDS", "STYLE", "TYPOGRAPHY"]:
        features[f"err_cat_{cat_name.lower()}"] = float(categories.get(cat_name, 0))

    confused_count = categories.get("CONFUSED_WORDS", 0)
    features["err_confused_words_count"] = float(confused_count)
    features["err_confused_words_density"] = (
        confused_count / word_count if word_count > 0 else 0.0
    )

    return features
