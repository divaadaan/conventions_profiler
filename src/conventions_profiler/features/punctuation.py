"""Punctuation features detecting sentence boundary failures."""

import statistics

from conventions_profiler.models import Essay


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    return statistics.mean(values) if values else default


def _count_lt_matches(matches: list[dict], rule_ids: set[str]) -> int:
    """Count LanguageTool matches for a set of rule IDs."""
    return sum(1 for m in matches if m.get("rule_id") in rule_ids)


def compute_punctuation_features(essay: Essay) -> dict[str, float]:
    """Extract punctuation features from spaCy and LanguageTool pass results."""
    spacy_data = essay.pass_results.get("spacy", {})
    lt_data = essay.pass_results.get("languagetool", {})
    sentences = spacy_data.get("sentences", [])
    lt_matches = lt_data.get("matches", [])
    total_tokens = spacy_data.get("total_tokens", 1)

    if not sentences:
        return _empty_features()

    features: dict[str, float] = {}

    # Punctuation density
    total_punct = sum(s["punct_count"] for s in sentences)
    features["punct_density"] = total_punct / total_tokens if total_tokens > 0 else 0.0

    # Punctuation to clause ratio
    total_clauses = sum(s["clauses_with_subjects"] for s in sentences)
    features["punct_to_clause_ratio"] = (
        total_punct / total_clauses if total_clauses > 0 else 0.0
    )

    # Comma splice / run-on detection via LanguageTool
    comma_splice_candidates = _count_lt_matches(
        lt_matches, {"COMMA_COMPOUND_SENTENCE", "COMMA_COMPOUND_SENTENCE_2"}
    )

    features["punct_comma_splice_candidates"] = float(comma_splice_candidates)
    features["punct_comma_splice_ratio"] = (
        comma_splice_candidates / len(sentences) if sentences else 0.0
    )

    # Sentence length outliers
    word_counts = [s["word_count"] for s in sentences]
    if len(word_counts) >= 2:
        mean_wc = statistics.mean(word_counts)
        std_wc = statistics.stdev(word_counts) if len(word_counts) >= 2 else 0
        threshold = mean_wc + 2 * std_wc if std_wc > 0 else mean_wc * 2

        outlier_count = sum(1 for wc in word_counts if wc > threshold)
        features["punct_length_outlier_count"] = float(outlier_count)
        features["punct_length_outlier_ratio"] = outlier_count / len(sentences)
    else:
        features["punct_length_outlier_count"] = 0.0
        features["punct_length_outlier_ratio"] = 0.0

    # Low-punctuation long sentences
    low_punct_long = 0
    for sent in sentences:
        if sent["word_count"] > 20:  # Reasonably long sentence
            punct_rate = sent["punct_count"] / sent["word_count"]
            if punct_rate < 0.03:  # Less than ~1 punct per 33 words
                low_punct_long += 1

    features["punct_low_punct_long_count"] = float(low_punct_long)
    features["punct_low_punct_long_ratio"] = (
        low_punct_long / len(sentences) if sentences else 0.0
    )

    return features


def _empty_features() -> dict[str, float]:
    keys = [
        "punct_density", "punct_to_clause_ratio",
        "punct_comma_splice_candidates", "punct_comma_splice_ratio",
        "punct_length_outlier_count", "punct_length_outlier_ratio",
        "punct_low_punct_long_count", "punct_low_punct_long_ratio",
    ]
    return {k: 0.0 for k in keys}
