"""Validation harness for surprisal features (Pythia 70M default).

Surprisal values vary with the LM, so bounds here are deliberately loose —
they assert plausible per-word-bits ranges and the *relative ordering*
between fluent and disfluent text. Tight calibration belongs in the
diagnostic CSV (``--report``), not the assertions.

Requires torch + transformers (skipped via importorskip in conftest).
"""

from __future__ import annotations

import math

import pytest

from feature_extractor.features.surprisal import compute_surprisal_features
from tests.harness.base import (
    FeatureCase,
    assert_ranges,
    run_feature_case,
    write_report,
)


CASES: list[FeatureCase] = [
    FeatureCase(
        id="surp_fluent_01",
        tier="clean",
        text=(
            "The cat sat on the mat. The dog ran in the park. "
            "We had fun all day."
        ),
        description="Short fluent sentences with common vocabulary.",
        expected_ranges={
            # Mean per-word surprisal under Pythia 70M for fluent simple
            # text is moderate (model is small, so absolute values are
            # higher than larger LMs would give).
            "surp_mean_per_word": (1.0, 15.0),
            # Perplexity = 2 ** mean ∈ a wide but bounded range.
            "surp_perplexity": (2.0, 32768.0),
            # Variability should be present but bounded.
            "surp_std_per_word": (0.0, 15.0),
            # P95 is at least the mean.
            "surp_p95_per_word": (1.0, 30.0),
        },
    ),
    FeatureCase(
        id="surp_garbled_01",
        tier="severe",
        text=(
            "xqz blorpf the ungainly meeple frumiously bunkpfed. "
            "Plonkward grambit, plonkward grambit, fnxqz!"
        ),
        description="Nonsense / OOV-heavy text. Expect very high mean surprisal.",
        expected_ranges={
            # Garbled text → higher mean than fluent baseline.
            "surp_mean_per_word": (5.0, 40.0),
            # Tail will spike on the OOV BPE pieces.
            "surp_p95_per_word": (8.0, 60.0),
        },
    ),
    FeatureCase(
        id="surp_single_word_01",
        tier="parse_distortion",
        text="Hello.",
        description="Single sentence, very few tokens — must not crash.",
        expected_ranges={
            # Some surprisal value; may be 0 if first token is dropped.
            "surp_mean_per_word": (0.0, 30.0),
        },
    ),
    FeatureCase(
        id="surp_empty_01",
        tier="parse_distortion",
        text="",
        description="Empty text — all aggregates default to zero.",
        expected_ranges={
            "surp_mean_per_word": (0.0, 0.0),
            "surp_perplexity": (0.0, 1.0),  # 2 ** 0 == 1, default is 0.0
            "surp_p95_per_word": (0.0, 0.0),
            "surp_std_per_word": (0.0, 0.0),
            "surp_sentence_mean_max": (0.0, 0.0),
            "surp_sentence_mean_std": (0.0, 0.0),
        },
    ),
]


def test_all_cases_produce_finite_features(spacy_pass, surprisal_pass):
    for case in CASES:
        _, features = run_feature_case(
            case,
            spacy_pass,
            compute_surprisal_features,
            extra_passes=[surprisal_pass],
        )
        for name, val in features.items():
            assert math.isfinite(val), f"[{case.id}] {name}={val} is not finite"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_surprisal_case(case, spacy_pass, surprisal_pass):
    _, features = run_feature_case(
        case,
        spacy_pass,
        compute_surprisal_features,
        extra_passes=[surprisal_pass],
    )
    failures = assert_ranges(case, features)
    if failures:
        pytest.fail("\n".join(failures))


def test_garbled_higher_than_fluent(spacy_pass, surprisal_pass):
    """Relative-ordering check: garbled text should have higher mean
    surprisal than fluent text. Far more robust than absolute bounds."""
    fluent = next(c for c in CASES if c.id == "surp_fluent_01")
    garbled = next(c for c in CASES if c.id == "surp_garbled_01")

    _, fluent_feats = run_feature_case(
        fluent, spacy_pass, compute_surprisal_features,
        extra_passes=[surprisal_pass],
    )
    _, garbled_feats = run_feature_case(
        garbled, spacy_pass, compute_surprisal_features,
        extra_passes=[surprisal_pass],
    )
    assert (
        garbled_feats["surp_mean_per_word"]
        > fluent_feats["surp_mean_per_word"]
    ), (
        f"Garbled mean {garbled_feats['surp_mean_per_word']:.3f} should exceed "
        f"fluent mean {fluent_feats['surp_mean_per_word']:.3f}"
    )


def test_diagnostic_report(spacy_pass, surprisal_pass, report_enabled):
    if not report_enabled:
        pytest.skip("--report not set")

    rows = []
    for case in CASES:
        _, features = run_feature_case(
            case,
            spacy_pass,
            compute_surprisal_features,
            extra_passes=[surprisal_pass],
        )
        rows.append({
            "case_id": case.id,
            "tier": case.tier,
            "description": case.description,
            "text": case.text,
            **{f"feat_{k}": v for k, v in features.items()},
        })
    out_path = write_report("surprisal", rows)
    print(f"\nDiagnostic report written to: {out_path}")
