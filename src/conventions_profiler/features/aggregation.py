"""
Document-level aggregation features.

Computes summary statistics across sentence-level features and identifies
patterns of error concentration. A document with errors concentrated in
a few sentences may be more interpretable than one with errors distributed
throughout — the reader can at least follow most of the text.
"""

import statistics

from conventions_profiler.models import Essay


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    return statistics.mean(values) if values else default


def _safe_stdev(values: list[float], default: float = 0.0) -> float:
    return statistics.stdev(values) if len(values) >= 2 else default


def compute_aggregation_features(essay: Essay) -> dict[str, float]:
    """Compute document-level aggregation features from pass results.

    These features capture patterns that individual sentence-level features
    miss: error concentration, worst-case windows, and document-wide consistency.
    """
    spacy_data = essay.pass_results.get("spacy", {})
    lt_data = essay.pass_results.get("languagetool", {})
    sentences = spacy_data.get("sentences", [])
    matches = lt_data.get("matches", [])

    features: dict[str, float] = {}

    # --- Document size features ---
    features["agg_num_sentences"] = float(len(sentences))
    features["agg_total_words"] = float(spacy_data.get("total_words", 0))
    features["agg_words_per_sentence"] = (
        features["agg_total_words"] / features["agg_num_sentences"]
        if features["agg_num_sentences"] > 0 else 0.0
    )

    # --- Errors per sentence ---
    # Distribute errors across sentences by offset
    errors_per_sentence = _distribute_errors_to_sentences(sentences, matches, essay.text)
    features["agg_errors_per_sentence_mean"] = _safe_mean(errors_per_sentence)
    features["agg_errors_per_sentence_max"] = (
        max(errors_per_sentence) if errors_per_sentence else 0.0
    )
    features["agg_errors_per_sentence_stdev"] = _safe_stdev(errors_per_sentence)

    # --- Proportion of error-free sentences ---
    error_free = sum(1 for e in errors_per_sentence if e == 0)
    features["agg_error_free_sentence_ratio"] = (
        error_free / len(errors_per_sentence) if errors_per_sentence else 0.0
    )

    # --- Worst-window analysis ---
    # Sliding window of 3 sentences: find the window with highest error density
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

    # --- MDD consistency ---
    # High MDD variance may indicate inconsistent control of syntax
    sentence_mdds = [s.get("mdd", 0.0) for s in sentences]
    features["agg_mdd_consistency"] = _safe_stdev(sentence_mdds)

    # --- Error concentration index ---
    # Gini-like measure: are errors spread evenly or concentrated?
    # High concentration (few sentences with many errors) may be less damaging
    # than uniform distribution (every sentence has errors)
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


def _distribute_errors_to_sentences(
    sentences: list[dict],
    matches: list[dict],
    essay_text: str,
) -> list[float]:
    """Map LanguageTool errors to sentences by character offset.

    Returns a list of error counts, one per sentence.
    """
    if not sentences:
        return []

    # Compute approximate character offsets for each sentence
    # by finding sentence text positions in the essay
    sent_ranges = []
    search_start = 0
    for sent in sentences:
        sent_text = sent.get("text", "")
        idx = essay_text.find(sent_text, search_start)
        if idx == -1:
            # Fallback: approximate position
            idx = search_start
        sent_ranges.append((idx, idx + len(sent_text)))
        search_start = idx + len(sent_text)

    # Count errors per sentence
    errors_per_sent = [0.0] * len(sentences)
    for match in matches:
        offset = match.get("offset", 0)
        for i, (start, end) in enumerate(sent_ranges):
            if start <= offset < end:
                errors_per_sent[i] += 1.0
                break

    return errors_per_sent
