"""Canonical feature-extraction package.

This is the single source of truth for turning raw keystroke events into the
timing-feature sequences consumed by the embedding model. Per CLAUDE.md, this
module must be imported by both the (future) training pipeline and the
backend inference/demo code -- never reimplemented separately in JS or in a
second Python copy.
"""

from .extract import extract_features

__all__ = ["extract_features"]
