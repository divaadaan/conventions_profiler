"""Syntactic complexity features"""

import statistics

from feature_extractor.models import Essay


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    return statistics.mean(values) if values else default


def _safe_stdev(values: list[float], default: float = 0.0) -> float:
    return statistics.stdev(values) if len(values) >= 2 else default


def compute_syntactic_complexity_features(essay: Essay) -> dict[str, float]:
    spacy_data = essay.pass_results.get("spacy", {})
    sentences = spacy_data.get("sentences", [])

    if not sentences:
        return _empty_features()

    features: dict[str, float] = {}

    word_counts = [s["word_count"] for s in sentences]
    features["syn_sentence_length_mean"] = _safe_mean(word_counts)
    features["syn_sentence_length_stdev"] = _safe_stdev(word_counts)
    features["syn_sentence_length_max"] = max(word_counts) if word_counts else 0.0

    clauses_per_sent = [float(s["clauses_with_subjects"]) for s in sentences]
    features["syn_clauses_per_sentence_mean"] = _safe_mean(clauses_per_sent)
    features["syn_clauses_per_sentence_max"] = max(clauses_per_sent) if clauses_per_sent else 0.0

    all_noun_dep_counts: list[float] = []
    for sent in sentences:
        all_noun_dep_counts.extend(sent.get("noun_dep_counts", []))

    features["syn_phrasal_elaboration_mean"] = _safe_mean(all_noun_dep_counts)

    sent_phrasal = [
        _safe_mean([float(x) for x in s.get("noun_dep_counts", [])])
        for s in sentences
    ]
    features["syn_phrasal_elaboration_stdev"] = _safe_stdev(sent_phrasal)

    phrasal_to_clausal = []
    for sent in sentences:
        wc = sent["word_count"]
        if wc > 0:
            mean_noun_deps = _safe_mean(
                [float(x) for x in sent.get("noun_dep_counts", [])]
            )
            phrasal_to_clausal.append(mean_noun_deps / wc)
        else:
            phrasal_to_clausal.append(0.0)

    features["syn_phrasal_clausal_ratio_mean"] = _safe_mean(phrasal_to_clausal)

    if len(word_counts) >= 2 and _safe_stdev(word_counts) > 0:
        mean_wc = _safe_mean(word_counts)
        std_wc = _safe_stdev(word_counts)
        threshold = mean_wc + 2 * std_wc

        long_low_elaboration = 0
        long_high_elaboration = 0

        for sent in sentences:
            if sent["word_count"] > threshold:
                phrasal = _safe_mean(
                    [float(x) for x in sent.get("noun_dep_counts", [])]
                )
                if phrasal < _safe_mean(all_noun_dep_counts):
                    long_low_elaboration += 1
                else:
                    long_high_elaboration += 1

        features["syn_long_low_elaboration_count"] = float(long_low_elaboration)
        features["syn_long_high_elaboration_count"] = float(long_high_elaboration)
    else:
        features["syn_long_low_elaboration_count"] = 0.0
        features["syn_long_high_elaboration_count"] = 0.0

    if len(word_counts) >= 2:
        features["syn_sentence_length_cv"] = (
            _safe_stdev(word_counts) / _safe_mean(word_counts)
            if _safe_mean(word_counts) > 0 else 0.0
        )
    else:
        features["syn_sentence_length_cv"] = 0.0

    n = len(sentences)
    type_counts: dict[str, int] = {
        "simple": 0, "compound": 0, "complex": 0, "compound_complex": 0
    }
    for s in sentences:
        st = s.get("sentence_type", "simple")
        if st in type_counts:
            type_counts[st] += 1
    features["syn_simple_sent_rate"] = type_counts["simple"] / n
    features["syn_compound_sent_rate"] = type_counts["compound"] / n
    features["syn_complex_sent_rate"] = type_counts["complex"] / n
    features["syn_compound_complex_sent_rate"] = type_counts["compound_complex"] / n

    return features


def _empty_features() -> dict[str, float]:
    keys = [
        "syn_sentence_length_mean", "syn_sentence_length_stdev",
        "syn_sentence_length_max",
        "syn_clauses_per_sentence_mean", "syn_clauses_per_sentence_max",
        "syn_phrasal_elaboration_mean", "syn_phrasal_elaboration_stdev",
        "syn_phrasal_clausal_ratio_mean",
        "syn_long_low_elaboration_count", "syn_long_high_elaboration_count",
        "syn_sentence_length_cv",
        "syn_simple_sent_rate", "syn_compound_sent_rate",
        "syn_complex_sent_rate", "syn_compound_complex_sent_rate",
    ]
    return {k: 0.0 for k in keys}
