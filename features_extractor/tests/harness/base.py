"""Shared scaffold for feature validation harnesses.

Each per-feature test file defines a list of `FeatureCase`s. The shared
runner here wraps text in an Essay, runs the requested passes, applies a
feature-module function, and asserts that each named feature lands in
its expected (low, high) range.

Diagnostic CSVs (one row per case x feature) are written to
``docs/reports/harness/<feature_set>.csv`` when the ``--report`` pytest
flag is set; otherwise the run is assertion-only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from feature_extractor.models import Essay
from feature_extractor.passes.base import BasePass
from feature_extractor.passes.languagetool_pass import LanguageToolPass
from feature_extractor.passes.spacy_pass import SpacyPass

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = REPO_ROOT / "docs" / "reports" / "harness"


@dataclass
class FeatureCase:
    """A single fixture with expected feature ranges.

    expected_ranges maps feature_name -> (low, high) inclusive bounds.
    Use float('-inf') / float('inf') for one-sided expectations.
    """
    id: str
    text: str
    description: str
    expected_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    tier: str = "isolated"  # "clean" | "isolated" | "co_occurring" | "severe" | "parse_distortion"


def _make_essay(case: FeatureCase) -> Essay:
    return Essay(essay_id=case.id, text=case.text, prompt_id="harness")


def run_feature_case(
    case: FeatureCase,
    spacy_pass: SpacyPass,
    feature_fn: Callable[[Essay], dict[str, float]],
    languagetool_pass: LanguageToolPass | None = None,
    extra_passes: list[BasePass] | None = None,
) -> tuple[Essay, dict[str, float]]:
    """Run the requested passes and feature module on a single case."""
    essay = _make_essay(case)
    spacy_pass.process_essay(essay)
    if languagetool_pass is not None:
        languagetool_pass.process_essay(essay)
    for p in extra_passes or ():
        p.process_essay(essay)
    features = feature_fn(essay)
    return essay, features


def assert_ranges(case: FeatureCase, features: dict[str, float]) -> list[str]:
    """Check feature values against expected_ranges. Returns list of failure messages."""
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
    return failures


def write_report(
    feature_set: str,
    rows: list[dict],
) -> Path:
    """Write a per-feature-set diagnostic CSV. Returns the output path."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"{feature_set}.csv"
    if not rows:
        return out_path
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path
