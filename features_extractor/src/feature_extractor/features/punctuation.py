"""Punctuation features detecting sentence boundary failures."""

import statistics

from feature_extractor.models import Essay


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    return statistics.mean(values) if values else default


def _count_lt_matches(matches: list[dict], rule_ids: set[str]) -> int:
    return sum(1 for m in matches if m.get("rule_id") in rule_ids)


def compute_punctuation_features(essay: Essay) -> dict[str, float]:
    spacy_data = essay.pass_results.get("spacy", {})
    lt_data = essay.pass_results.get("languagetool", {})
    sentences = spacy_data.get("sentences", [])
    lt_matches = lt_data.get("matches", [])
    total_tokens = spacy_data.get("total_tokens", 1)

    if not sentences:
        return _empty_features()

    features: dict[str, float] = {}

    total_punct = sum(s["punct_count"] for s in sentences)
    features["punct_density"] = total_punct / total_tokens if total_tokens > 0 else 0.0

    total_clauses = sum(s["clauses_with_subjects"] for s in sentences)
    features["punct_to_clause_ratio"] = (
        total_punct / total_clauses if total_clauses > 0 else 0.0
    )

    comma_splice_candidates = _count_lt_matches(
        lt_matches, {"COMMA_COMPOUND_SENTENCE", "COMMA_COMPOUND_SENTENCE_2"}
    )

    features["punct_comma_splice_candidates"] = float(comma_splice_candidates)
    comma_splice_ratio = (
        comma_splice_candidates / len(sentences) if sentences else 0.0
    )
    features["punct_comma_splice_ratio"] = comma_splice_ratio
    # Ratio can exceed 1 (multiple flags per sentence), clamp before flipping.
    features["punct_comma_splice_correct_rate"] = max(
        0.0, 1.0 - min(comma_splice_ratio, 1.0)
    )

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

    low_punct_long = 0
    for sent in sentences:
        if sent["word_count"] > 20:
            punct_rate = sent["punct_count"] / sent["word_count"]
            if punct_rate < 0.03:
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
    empty = {k: 0.0 for k in keys}
    empty["punct_comma_splice_correct_rate"] = 1.0
    return empty
