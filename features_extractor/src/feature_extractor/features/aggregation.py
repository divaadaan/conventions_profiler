"""Document-level aggregation features"""

import statistics

from feature_extractor.features._helpers import distribute_errors_to_sentences
from feature_extractor.models import Essay


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    return statistics.mean(values) if values else default


def _safe_stdev(values: list[float], default: float = 0.0) -> float:
    return statistics.stdev(values) if len(values) >= 2 else default


def compute_aggregation_features(essay: Essay) -> dict[str, float]:
    spacy_data = essay.pass_results.get("spacy", {})
    lt_data = essay.pass_results.get("languagetool", {})
    sentences = spacy_data.get("sentences", [])
    matches = lt_data.get("matches", [])

    features: dict[str, float] = {}

    features["agg_num_sentences"] = float(len(sentences))
    features["agg_total_words"] = float(spacy_data.get("total_words", 0))
    features["agg_words_per_sentence"] = (
        features["agg_total_words"] / features["agg_num_sentences"]
        if features["agg_num_sentences"] > 0 else 0.0
    )

    errors_per_sentence = distribute_errors_to_sentences(sentences, matches)
    features["agg_errors_per_sentence_mean"] = _safe_mean(errors_per_sentence)
    features["agg_errors_per_sentence_max"] = (
        max(errors_per_sentence) if errors_per_sentence else 0.0
    )
    features["agg_errors_per_sentence_stdev"] = _safe_stdev(errors_per_sentence)

    error_free = sum(1 for e in errors_per_sentence if e == 0)
    features["agg_error_free_sentence_ratio"] = (
        error_free / len(errors_per_sentence) if errors_per_sentence else 0.0
    )

    if len(errors_per_sentence) >= 3:
        window_sums = [
            sum(errors_per_sentence[i:i+3])
            for i in range(len(errors_per_sentence) - 2)
        ]
        features["agg_worst_window_3_errors"] = float(max(window_sums))
        features["agg_worst_window_3_mean"] = _safe_mean(
            [float(s) for s in window_sums]
        )
    else:
        features["agg_worst_window_3_errors"] = float(sum(errors_per_sentence))
        features["agg_worst_window_3_mean"] = _safe_mean(
            [float(e) for e in errors_per_sentence]
        )

    sentence_mdds = [s.get("mdd", 0.0) for s in sentences]
    features["agg_mdd_consistency"] = _safe_stdev(sentence_mdds)

    if errors_per_sentence and sum(errors_per_sentence) > 0:
        total = sum(errors_per_sentence)
        n = len(errors_per_sentence)
        sorted_errors = sorted(errors_per_sentence)
        cumulative = [sum(sorted_errors[:i+1]) / total for i in range(n)]
        expected = [(i + 1) / n for i in range(n)]
        gini = sum(e - c for e, c in zip(expected, cumulative)) / n
        features["agg_error_concentration"] = gini
    else:
        features["agg_error_concentration"] = 0.0

    return features
