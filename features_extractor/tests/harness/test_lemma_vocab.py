"""Validation harness for lemma-based vocabulary features."""

from __future__ import annotations

import math

import pytest

from feature_extractor.features.lemma_vocab import compute_lemma_vocab_features
from tests.harness.base import (
    FeatureCase,
    assert_ranges,
    run_feature_case,
    write_report,
)

CASES: list[FeatureCase] = [
    FeatureCase(
        id="lemma_diverse_01",
        tier="clean",
        text=(
            "Children explore vibrant gardens. "
            "Curious teachers prepare engaging lessons. "
            "Distant mountains appear majestic at sunrise."
        ),
        description="High lexical diversity — most lemmas appear once.",
        expected_ranges={
            # High hapax count expected (most words unique)
            "vocab_unique_lemma_count": (8.0, 30.0),
            # High type-token ratio (unique lemmas / total tokens)
            "vocab_lemma_diversity_ratio": (0.7, 1.0),
        },
    ),
    FeatureCase(
        id="lemma_repetitive_01",
        tier="isolated",
        text=(
            "I run and I run and I run. "
            "I run because I run. "
            "Running runs run."  # lemma 'run' across all variants
        ),
        description="High repetition — 'run' surface variants collapse to one lemma.",
        expected_ranges={
            # Low diversity: many tokens, few unique lemmas
            "vocab_lemma_diversity_ratio": (0.0, 0.6),
        },
    ),
    FeatureCase(
        id="lemma_inflection_dedup_01",
        tier="isolated",
        text="The runner runs and ran. He has been running. She runs too.",
        description="Inflected variants of 'run' should dedupe to one lemma.",
        expected_ranges={
            # We expect lemma diversity ratio < surface-form TTR; can't assert
            # exact value, but should be moderate.
            "vocab_lemma_diversity_ratio": (0.4, 0.95),
        },
    ),
    FeatureCase(
        id="lemma_empty_01",
        tier="parse_distortion",
        text="...",
        description="Punctuation-only — no alpha tokens. Returns zeros.",
        expected_ranges={
            "vocab_unique_lemma_count": (0.0, 0.0),
            "vocab_lemma_diversity_ratio": (0.0, 0.0),
        },
    ),
]


def test_all_cases_produce_finite_features(spacy_pass):
    for case in CASES:
        _, features = run_feature_case(case, spacy_pass, compute_lemma_vocab_features)
        for name, val in features.items():
            assert math.isfinite(val), f"[{case.id}] {name}={val} is not finite"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_lemma_vocab_case(case, spacy_pass):
    _, features = run_feature_case(case, spacy_pass, compute_lemma_vocab_features)
    failures = assert_ranges(case, features)
    if failures:
        pytest.fail("\n".join(failures))


def test_diagnostic_report(spacy_pass, report_enabled):
    if not report_enabled:
        pytest.skip("--report not set")

    rows = []
    for case in CASES:
        _, features = run_feature_case(case, spacy_pass, compute_lemma_vocab_features)
        rows.append({
            "case_id": case.id,
            "tier": case.tier,
            "description": case.description,
            "text": case.text,
            **{f"feat_{k}": v for k, v in features.items()},
        })
    out_path = write_report("lemma_vocab", rows)
    print(f"\nDiagnostic report written to: {out_path}")
