"""Shared helpers for feature modules."""

from __future__ import annotations


def distribute_errors_to_sentences(
    sentences: list[dict],
    matches: list[dict],
) -> list[float]:
    """Map LanguageTool match offsets to sentence indices.

    Returns a list of error counts, one per sentence (in order).
    """
    if not sentences:
        return []

    sent_ranges = [(s["start_char"], s["end_char"]) for s in sentences]
    counts = [0.0] * len(sentences)
    for match in matches:
        offset = match.get("offset", 0)
        for i, (start, end) in enumerate(sent_ranges):
            if start <= offset < end:
                counts[i] += 1.0
                break
    return counts
