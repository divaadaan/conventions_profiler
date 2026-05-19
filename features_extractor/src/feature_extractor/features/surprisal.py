"""Surprisal-derived features.

Consumes per-word surprisal from SurprisalPass and sentence boundaries from
the spaCy pass; emits essay-level aggregates. Words with surprisal=None
(e.g., document-initial) are excluded from aggregates.
"""

from __future__ import annotations

import numpy as np

from feature_extractor.models import Essay


_FEATURE_NAMES = (
    "surp_mean_per_word",
    "surp_perplexity",
    "surp_p95_per_word",
    "surp_std_per_word",
    "surp_sentence_mean_max",
    "surp_sentence_mean_std",
)


def _empty_features() -> dict[str, float]:
    return {name: 0.0 for name in _FEATURE_NAMES}


def compute_surprisal_features(essay: Essay) -> dict[str, float]:
    surp_data = essay.pass_results.get("surprisal")
    if not isinstance(surp_data, dict):
        return _empty_features()

    words = surp_data.get("words") or []
    scored = [w for w in words if w.get("surprisal") is not None]

    if not scored:
        return _empty_features()

    surp_values = np.array([float(w["surprisal"]) for w in scored], dtype=np.float64)
    mean_surp = float(surp_values.mean())
    perplexity = float(2.0 ** mean_surp)
    p95 = float(np.percentile(surp_values, 95))
    # ddof=0 (population std) to avoid NaN when only one scored word exists.
    std_surp = float(surp_values.std(ddof=0))

    sentence_means = _per_sentence_means(essay, scored)
    if sentence_means:
        sm = np.array(sentence_means, dtype=np.float64)
        sentence_mean_max = float(sm.max())
        sentence_mean_std = float(sm.std(ddof=0))
    else:
        sentence_mean_max = mean_surp
        sentence_mean_std = 0.0

    return {
        "surp_mean_per_word": mean_surp,
        "surp_perplexity": perplexity,
        "surp_p95_per_word": p95,
        "surp_std_per_word": std_surp,
        "surp_sentence_mean_max": sentence_mean_max,
        "surp_sentence_mean_std": sentence_mean_std,
    }


def _per_sentence_means(essay: Essay, scored_words: list[dict]) -> list[float]:
    sentences = (
        essay.pass_results.get("spacy", {}).get("sentences") or []
    )
    if not sentences:
        vals = [float(w["surprisal"]) for w in scored_words]
        return [float(np.mean(vals))] if vals else []

    buckets: list[list[float]] = [[] for _ in sentences]
    for w in scored_words:
        w_start = w["start_char"]
        w_end = w["end_char"]
        for i, sent in enumerate(sentences):
            s_start = sent.get("start_char", 0)
            s_end = sent.get("end_char", 0)
            if s_start <= w_start < s_end:
                buckets[i].append(float(w["surprisal"]))
                break
        else:
            # Past all sentence ranges (rare) — fall into the last bucket.
            if w_end <= sentences[-1].get("end_char", 0) + 1:
                buckets[-1].append(float(w["surprisal"]))

    return [float(np.mean(b)) for b in buckets if b]
