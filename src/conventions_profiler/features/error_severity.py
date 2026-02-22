"""
Error severity features implementing the global/local error framework.

Maps LanguageTool rule categories and rule IDs to severity tiers based on
Burt & Kiparsky (1974): global errors block comprehension, local errors
are cosmetic. Severity weights are configurable for later tuning.

The severity tiers and weights are starting points — they become
hyperparameters when the client trains on their operational data.
"""

from conventions_profiler.models import Essay

# --- Severity tier mappings ---
# These map LanguageTool category names and specific rule IDs to severity tiers.
# Categories not listed here default to MEDIUM.

GLOBAL_CATEGORIES = {
    # Structural errors that block sentence-level comprehension
    "SENTENCE_FRAGMENT",
    "WORD_ORDER",
    "MISSING_VERB",
}

GLOBAL_RULES = {
    # Specific rule IDs that are comprehension-blocking even if their
    # category is not globally severe
    "SENTENCE_FRAGMENT",
    "MISSING_VERB_BEFORE",
    "MISSING_VERB_AFTER",
    "WORD_ORDER_ENGLISH",
    "CONFUSION_RULE",  # Real-word errors that change meaning (there/their)
    "COMMA_SPLICE",
    "RUN_ON_SENTENCE",
    "FUSED_SENTENCE",
    "DOUBLE_NEGATIVE",
}

LOCAL_CATEGORIES = {
    # Cosmetic errors that don't impede comprehension
    "TYPOS",
    "CASING",
    "COMPOUNDING",
    "REDUNDANCY",
    "STYLE",
}

LOCAL_RULES = {
    "UPPERCASE_SENTENCE_START",
    "EN_UNPAIRED_BRACKETS",
    "WHITESPACE_RULE",
    "EN_QUOTES",
    "COMMA_PARENTHESIS_WHITESPACE",
}

# Default severity weights
SEVERITY_WEIGHTS = {
    "global": 3.0,
    "medium": 2.0,
    "local": 1.0,
}


def _classify_severity(match: dict) -> str:
    """Classify a single LanguageTool match into a severity tier."""
    rule_id = match.get("rule_id", "")
    category = match.get("category", "")

    # Check specific rules first (more precise)
    if rule_id in GLOBAL_RULES:
        return "global"
    if rule_id in LOCAL_RULES:
        return "local"

    # Then check categories
    if category in GLOBAL_CATEGORIES:
        return "global"
    if category in LOCAL_CATEGORIES:
        return "local"

    # Default to medium
    return "medium"


def compute_error_severity_features(essay: Essay) -> dict[str, float]:
    """Extract error severity features from LanguageTool pass results."""
    lt_data = essay.pass_results.get("languagetool", {})
    matches = lt_data.get("matches", [])
    word_count = essay.word_count

    features: dict[str, float] = {}

    # --- Raw error counts ---
    features["err_total_count"] = float(len(matches))
    features["err_total_density"] = len(matches) / word_count if word_count > 0 else 0.0

    # --- Classify and count by severity ---
    severity_counts = {"global": 0, "medium": 0, "local": 0}
    for match in matches:
        tier = _classify_severity(match)
        severity_counts[tier] += 1

    features["err_global_count"] = float(severity_counts["global"])
    features["err_medium_count"] = float(severity_counts["medium"])
    features["err_local_count"] = float(severity_counts["local"])

    # --- Severity densities (per word) ---
    features["err_global_density"] = (
        severity_counts["global"] / word_count if word_count > 0 else 0.0
    )
    features["err_medium_density"] = (
        severity_counts["medium"] / word_count if word_count > 0 else 0.0
    )
    features["err_local_density"] = (
        severity_counts["local"] / word_count if word_count > 0 else 0.0
    )

    # --- Weighted error density ---
    # The key aggregate feature: total error burden weighted by severity
    weighted_sum = sum(
        severity_counts[tier] * SEVERITY_WEIGHTS[tier]
        for tier in severity_counts
    )
    features["err_weighted_density"] = weighted_sum / word_count if word_count > 0 else 0.0

    # --- Global error ratio ---
    # Proportion of errors that are comprehension-blocking
    # High ratio = even moderate error density causes communication failure
    total_errors = len(matches)
    features["err_global_ratio"] = (
        severity_counts["global"] / total_errors if total_errors > 0 else 0.0
    )

    # --- Error category distribution ---
    # Counts by LanguageTool category (top categories)
    categories = lt_data.get("categories", {})
    for cat_name in ["GRAMMAR", "TYPOS", "PUNCTUATION", "CASING",
                     "CONFUSED_WORDS", "STYLE", "TYPOGRAPHY"]:
        features[f"err_cat_{cat_name.lower()}"] = float(categories.get(cat_name, 0))

    # --- Confused words (real-word errors) ---
    # These are particularly damaging because context can't resolve them
    confused_count = categories.get("CONFUSED_WORDS", 0)
    features["err_confused_words_count"] = float(confused_count)
    features["err_confused_words_density"] = (
        confused_count / word_count if word_count > 0 else 0.0
    )

    return features
