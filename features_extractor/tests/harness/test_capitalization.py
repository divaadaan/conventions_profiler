"""Validation harness for capitalization features.

Cases are organized into severity tiers:
    clean             baseline with no capitalization errors
    isolated          single targeted error
    co_occurring      target error alongside unrelated errors
    severe            many targeted errors at high density
    parse_distortion  malformed text — feature must not crash or produce NaN

Each case asserts expected ranges per feature. Run with --report to dump
diagnostic CSVs.
"""

from __future__ import annotations

import math

import pytest

from feature_extractor.features.capitalization import compute_capitalization_features
from tests.harness.base import (
    FeatureCase,
    assert_ranges,
    run_feature_case,
    write_report,
)

CASES: list[FeatureCase] = [
    # ---------------- CLEAN BASELINE ----------------
    FeatureCase(
        id="cap_clean_01",
        tier="clean",
        text=(
            "The students walked to the park on Saturday. "
            "Maria and James met them near the entrance. "
            "They played soccer until the sun went down."
        ),
        description="Three clean sentences, proper nouns capitalized, no errors.",
        expected_ranges={
            "cap_sentence_initial_rate": (0.0, 0.0),
            "cap_proper_noun_rate": (0.0, 0.0),
            "cap_intrusion_rate": (0.0, 0.0),
        },
    ),

    # ---------------- ISOLATED: SENTENCE-INITIAL ----------------
    FeatureCase(
        id="cap_sent_initial_01",
        tier="isolated",
        text=(
            "The students walked to the park. "
            "they played soccer for an hour. "
            "the game ended at five."
        ),
        description="Two of three sentences start with lowercase.",
        expected_ranges={
            "cap_sentence_initial_rate": (0.6, 0.7),  # 2/3
            "cap_proper_noun_rate": (0.0, 0.0),
            "cap_intrusion_rate": (0.0, 0.0),
        },
    ),

    # ---------------- ISOLATED: PROPER NOUN ----------------
    FeatureCase(
        id="cap_proper_noun_01",
        tier="isolated",
        text=(
            "Last summer my family travelled to toronto. "
            "We met aunt sarah at the airport. "
            "She had flown in from vancouver that morning."
        ),
        description="Three proper nouns lowercased: toronto, sarah, vancouver.",
        expected_ranges={
            "cap_sentence_initial_rate": (0.0, 0.0),
            "cap_proper_noun_rate": (0.5, 1.0),
            "cap_intrusion_rate": (0.0, 0.0),
        },
    ),

    # ---------------- ISOLATED: MID-SENTENCE INTRUSION ----------------
    FeatureCase(
        id="cap_intrusion_01",
        tier="isolated",
        text=(
            "The students walked to the Park on saturday. "
            "They played Soccer until the sun went down."
        ),
        description='Two unexpected mid-sentence capitals: "Park" and "Soccer".',
        expected_ranges={
            "cap_sentence_initial_rate": (0.0, 0.0),
            "cap_intrusion_rate": (0.9, 1.1),
        },
    ),

    # ---------------- CO-OCCURRING ----------------
    FeatureCase(
        id="cap_cooccurring_01",
        tier="co_occurring",
        text=(
            "the students walked to the park on Saturday. "
            "they met aunt sarah near the entrance. "
            "She had been waiting since Noon."
        ),
        description="Two sentence-initial errors + one proper-noun error + one intrusion.",
        expected_ranges={
            "cap_sentence_initial_rate": (0.6, 0.7),  # 2/3
            "cap_proper_noun_rate": (0.9, 1.0),
            "cap_intrusion_rate": (0.3, 0.4),  # 1/3 ('Noon')
        },
    ),

    # ---------------- SEVERE ----------------
    FeatureCase(
        id="cap_severe_01",
        tier="severe",
        text=(
            "the students walked to the park on saturday. "
            "they met aunt sarah near the entrance. "
            "they played soccer until the sun went down. "
            "later, they all went home for dinner."
        ),
        description="All four sentences lowercase-initial; multiple lowercased proper nouns.",
        expected_ranges={
            "cap_sentence_initial_rate": (1.0, 1.0),
            "cap_proper_noun_rate": (0.4, 1.0),
            "cap_intrusion_rate": (0.0, 0.0),
        },
    ),

    # ---------------- PARSE DISTORTION ----------------
    FeatureCase(
        id="cap_parse_distortion_01",
        tier="parse_distortion",
        text=(
            "the dog ran fast the cat sat still no punctuation here words just keep going "
            "and going without breaks i wonder if spacy can even segment this"
        ),
        description="Run-on with no punctuation. Feature must produce finite values without crashing.",
        expected_ranges={
            "cap_sentence_initial_rate": (0.0, 1.0),
            "cap_proper_noun_rate": (0.0, 1.0),
            "cap_intrusion_rate": (0.0, 1.0),
        },
    ),

    # ---------------- EDGE: ACRONYMS / "I" ----------------
    FeatureCase(
        id="cap_edge_acronyms_01",
        tier="isolated",
        text=(
            "I visited NASA last summer. "
            "The USA has many landmarks. "
            "I learned about NATO at school."
        ),
        description='Acronyms (NASA, USA, NATO) and pronoun "I" must not trigger intrusion.',
        expected_ranges={
            "cap_sentence_initial_rate": (0.0, 0.0),
            "cap_intrusion_rate": (0.0, 0.0),
        },
    ),
]


def test_all_cases_produce_finite_features(spacy_pass):
    """every case must produce finite values for all three features."""
    rows = []
    for case in CASES:
        _, features = run_feature_case(case, spacy_pass, compute_capitalization_features)
        for name, val in features.items():
            assert math.isfinite(val), f"[{case.id}] {name}={val} is not finite"
            assert 0.0 <= val <= 1.0, f"[{case.id}] {name}={val} outside [0,1]"
        rows.append({"case_id": case.id, "tier": case.tier, **features})


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_capitalization_case(case, spacy_pass):
    _, features = run_feature_case(case, spacy_pass, compute_capitalization_features)
    failures = assert_ranges(case, features)
    if failures:
        pytest.fail("\n".join(failures))


def test_diagnostic_report(spacy_pass, report_enabled):
    """Optional: dump per-case feature values to docs/reports/harness/capitalization.csv."""
    if not report_enabled:
        pytest.skip("--report not set")

    rows = []
    for case in CASES:
        _, features = run_feature_case(case, spacy_pass, compute_capitalization_features)
        rows.append({
            "case_id": case.id,
            "tier": case.tier,
            "description": case.description,
            "text": case.text,
            **{f"feat_{k}": v for k, v in features.items()},
            **{f"expected_{k}": f"[{lo}, {hi}]" for k, (lo, hi) in case.expected_ranges.items()},
        })
    out_path = write_report("capitalization", rows)
    print(f"\nDiagnostic report written to: {out_path}")
