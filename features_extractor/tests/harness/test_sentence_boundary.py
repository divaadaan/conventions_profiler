"""Validation harness for sentence-boundary features."""

from __future__ import annotations

import math

import pytest

from feature_extractor.features.sentence_boundary import compute_sentence_boundary_features
from tests.harness.base import (
    FeatureCase,
    assert_ranges,
    run_feature_case,
    write_report,
)

CASES: list[FeatureCase] = [
    FeatureCase(
        id="bound_clean_01",
        tier="clean",
        text=(
            "The students walked to the park. "
            "They played soccer for an hour. "
            "The game ended at five."
        ),
        description="Three clean simple sentences with terminal punctuation.",
        expected_ranges={
            "bound_missing_term_punct_count": (0.0, 0.0),
            "bound_missing_term_punct_rate": (0.0, 0.0),
            "bound_boundary_error_rate": (0.0, 0.2),
        },
    ),

    FeatureCase(
        id="bound_missing_term_punct_01",
        tier="isolated",
        text=(
            "The students walked to the park "
            "They played soccer for an hour "
            "The game ended at five"
        ),
        description="No terminal punctuation anywhere.",
        expected_ranges={
            "bound_boundary_error_rate": (0.3, 5.0),
        },
    ),

    FeatureCase(
        id="bound_comma_splice_01",
        tier="isolated",
        text=(
            "The students walked to the park, they played soccer. "
            "The teacher waited near the gate."
        ),
        description="One comma splice in two-sentence essay.",
        expected_ranges={
            "bound_boundary_error_rate": (0.3, 2.0),
        },
    ),

    FeatureCase(
        id="bound_severe_01",
        tier="severe",
        text=(
            "the dog ran fast the cat sat still "
            "they all went home it was getting late "
            "no one said a word"
        ),
        description="Multiple boundary issues: missing punct + missing capitals.",
        expected_ranges={
            "bound_boundary_error_rate": (0.5, 5.0),
        },
    ),

    FeatureCase(
        id="bound_parse_distortion_01",
        tier="parse_distortion",
        text="and and and the the the running running running",
        description="Word salad — feature must produce finite values without crashing.",
        expected_ranges={
            "bound_missing_term_punct_rate": (0.0, 1.0),
            "bound_boundary_error_rate": (0.0, 10.0),
        },
    ),
]


def test_all_cases_produce_finite_features(spacy_pass, languagetool_pass):
    for case in CASES:
        _, features = run_feature_case(
            case, spacy_pass, compute_sentence_boundary_features,
            languagetool_pass=languagetool_pass,
        )
        for name, val in features.items():
            assert math.isfinite(val), f"[{case.id}] {name}={val} is not finite"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_sentence_boundary_case(case, spacy_pass, languagetool_pass):
    _, features = run_feature_case(
        case, spacy_pass, compute_sentence_boundary_features,
        languagetool_pass=languagetool_pass,
    )
    failures = assert_ranges(case, features)
    if failures:
        pytest.fail("\n".join(failures))


def test_diagnostic_report(spacy_pass, languagetool_pass, report_enabled):
    if not report_enabled:
        pytest.skip("--report not set")

    rows = []
    for case in CASES:
        _, features = run_feature_case(
            case, spacy_pass, compute_sentence_boundary_features,
            languagetool_pass=languagetool_pass,
        )
        rows.append({
            "case_id": case.id,
            "tier": case.tier,
            "description": case.description,
            "text": case.text,
            **{f"feat_{k}": v for k, v in features.items()},
        })
    out_path = write_report("sentence_boundary", rows)
    print(f"\nDiagnostic report written to: {out_path}")
