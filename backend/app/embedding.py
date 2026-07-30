"""Stub wrapping the future frozen embedding model f().

Per CLAUDE.md: training produces a single frozen function
f(keystroke event sequence) -> embedding. Everything downstream (enrollment,
identification, verification) is just a forward pass through this function
plus vector comparison. That model doesn't exist yet -- it depends on the
Aalto dataset training run happening separately -- so this stub raises rather
than fabricating embeddings.
"""

from typing import Any


def embed(features: "Any") -> "np.ndarray":  # noqa: F821
    """Run the frozen embedding network on an extracted feature sequence.

    Parameters
    ----------
    features:
        The windowed timing-feature sequence produced by
        ``features.extract.extract_features``.

    Returns
    -------
    np.ndarray
        Intended eventual return: a fixed-length (e.g. 128-dim), L2-normalized
        embedding vector.

    Notes
    -----
    Not implemented yet -- no trained model exists. See CLAUDE.md's "Model"
    section.
    """
    raise NotImplementedError("model not yet trained")
