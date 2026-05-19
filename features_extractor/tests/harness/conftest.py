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
