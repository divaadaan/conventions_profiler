"""Validation harness for readability features."""

from __future__ import annotations

import math

import pytest

from feature_extractor.features.readability import compute_readability_features
from feature_extractor.models import Essay
from tests.harness.base import FeatureCase, write_report

CASES: list[FeatureCase] = [
    FeatureCase(
        id="read_easy_01",
        tier="clean",
        text=(
            "The cat sat on the mat. "
            "The dog ran in the park. "
            "We had fun all day."
        ),
        description="Very simple short sentences and one-syllable words.",
        expected_ranges={
            "read_flesch_reading_ease": (60.0, 130.0),
            "read_flesch_kincaid_grade": (-5.0, 5.0),
            "read_dale_chall_difficult_word_density": (0.0, 0.2),
        },
    ),
    FeatureCase(
        id="read_hard_01",
        tier="clean",
        text=(
            "The pharmaceutical industry's regulatory environment, particularly with respect to "
            "biopharmaceutical innovation and intellectual-property exclusivity, increasingly "
            "depends on transnational harmonization initiatives. Notwithstanding the proliferation "
            "of bilateral agreements, jurisdictional discrepancies in patentability requirements "
            "continue to impede commercialization."
        ),
        description="Dense academic register with polysyllabic vocabulary.",
        expected_ranges={
            "read_flesch_reading_ease": (-150.0, 50.0),
            "read_flesch_kincaid_grade": (10.0, 40.0),
            "read_dale_chall_difficult_word_density": (0.3, 1.0),
        },
    ),
    FeatureCase(
        id="read_empty_01",
        tier="parse_distortion",
        text="",
        description="Empty text — must not crash, returns zeros.",
        expected_ranges={
            "read_flesch_reading_ease": (0.0, 0.0),
            "read_flesch_kincaid_grade": (0.0, 0.0),
            "read_dale_chall_difficult_word_density": (0.0, 0.0),
        },
    ),
]


def _features_from_text(case: FeatureCase) -> dict[str, float]:
    essay = Essay(essay_id=case.id, text=case.text, prompt_id="harness")
    return compute_readability_features(essay)


def test_all_cases_produce_finite_features():
    for case in CASES:
        features = _features_from_text(case)
        for name, val in features.items():
            assert math.isfinite(val), f"[{case.id}] {name}={val} is not finite"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_readability_case(case):
    features = _features_from_text(case)
    failures = []
    for feat_name, (lo, hi) in case.expected_ranges.items():
        if feat_name not in features:
            failures.append(f"[{case.id}] feature '{feat_name}' missing from output")
            continue
        val = features[feat_name]
        if not (lo <= val <= hi):
            failures.append(
                f"[{case.id}] {feat_name}={val:.4f} outside expected [{lo}, {hi}] "
                f"({case.description})"
            )
    if failures:
        pytest.fail("\n".join(failures))


def test_diagnostic_report(report_enabled):
    if not report_enabled:
        pytest.skip("--report not set")

    rows = []
    for case in CASES:
        features = _features_from_text(case)
        rows.append({
            "case_id": case.id,
            "tier": case.tier,
            "description": case.description,
            "text": case.text,
            **{f"feat_{k}": v for k, v in features.items()},
        })
    out_path = write_report("readability", rows)
    print(f"\nDiagnostic report written to: {out_path}")
