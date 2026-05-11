"""Pytest configuration for the feature validation harness.

Shared fixtures load spaCy and LanguageTool once per test session to keep
runtime reasonable. The ``--report`` flag triggers diagnostic CSV output
to ``docs/reports/harness/``.
"""

from __future__ import annotations

import os

import pytest

from feature_extractor.passes.languagetool_pass import LanguageToolPass
from feature_extractor.passes.spacy_pass import SpacyPass


def pytest_addoption(parser):
    parser.addoption(
        "--report",
        action="store_true",
        default=False,
        help="Write diagnostic CSV reports under docs/reports/harness/",
    )


@pytest.fixture(scope="session")
def report_enabled(request) -> bool:
    return request.config.getoption("--report")


@pytest.fixture(scope="session")
def spacy_pass() -> SpacyPass:
    """Session-scoped spaCy pass.

    Defaults to en_core_web_trf to match production parse fidelity.
    Override with HARNESS_SPACY_MODEL=en_core_web_sm for faster local runs
    (some cases may then become flaky — see harness notes).
    """
    model = os.environ.get("HARNESS_SPACY_MODEL", "en_core_web_trf")
    config = {"models": {"spacy": {"model": model}}, "hardware": {"batch_size": 8}}
    pass_ = SpacyPass(config)
    pass_.load_model()
    yield pass_
    pass_.unload_model()


@pytest.fixture(scope="session")
def languagetool_pass() -> LanguageToolPass:
    config = {"models": {"languagetool": {"language": "en-CA"}}}
    pass_ = LanguageToolPass(config)
    pass_.load_model()
    yield pass_
    pass_.unload_model()


@pytest.fixture(scope="session")
def surprisal_pass():
    """Session-scoped surprisal pass loaded with Pythia 70M on CPU.

    Skipped if torch/transformers aren't installed. The model download
    (~150MB) is cached by HF Hub on first use.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from feature_extractor.passes.surprisal_pass import SurprisalPass

    config = {
        "models": {"surprisal": {"model": "EleutherAI/pythia-70m"}},
        "hardware": {"device": "cpu"},
    }
    pass_ = SurprisalPass(config)
    pass_.load_model()
    yield pass_
    pass_.unload_model()
