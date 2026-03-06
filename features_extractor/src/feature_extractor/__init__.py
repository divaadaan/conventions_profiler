"""
feature-extractor
==================

Linguistic feature extraction pipeline for conventions scoring in AES.

The pipeline extracts features that measure how writing conventions errors
impact the interpretability of student essays, organized around two
processing passes (spaCy and LanguageTool) with pure-function feature
modules that operate on pass outputs.
"""

__version__ = "0.1.0"
