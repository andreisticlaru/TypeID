"""Canonical feature extraction: raw keystroke events -> timing-feature sequence.

This module is the ONE implementation of feature extraction referenced throughout
CLAUDE.md ("Route live browser events to a backend endpoint that runs the one
canonical Python feature-extraction function"). It must be imported by both the
training pipeline (operating on rows from the Aalto dataset) and the backend
/enroll and /identify endpoints (operating on live browser-captured events) --
never duplicated or reimplemented in JS.
"""

from __future__ import annotations

from typing import Any


def extract_features(events: list[dict[str, Any]]) -> "np.ndarray":  # noqa: F821
    """Convert a raw keydown/keyup event log into a windowed timing-feature sequence.

    Parameters
    ----------
    events:
        A list of raw keystroke events, each shaped like
        ``{"key": str, "code": str, "type": "keydown" | "keyup", "t": float}``,
        with ``t`` a millisecond timestamp (e.g. ``performance.now()`` from the
        browser, or the equivalent column in the Aalto dataset). This is exactly
        the event log captured by the frontend's ``onKeyEvent`` handler
        (see ``frontend/app.js``) and is the same shape used for Aalto rows once
        that dataset is loaded, so this function is the single mismatch point
        between training and live capture -- keep it that way.

    Returns
    -------
    np.ndarray
        Intended eventual return: a ``(M, 4)`` array of per-keystroke timing
        vectors ``(HL, IL, PL, RL)`` -- hold/inter-key/press/release latency --
        pooled or windowed to a fixed length ``M=50`` per CLAUDE.md's Data
        section, with zero-padding + masking for sequences shorter than the
        window. Backspace keystrokes are kept in the sequence rather than
        deleted, per CLAUDE.md's documented default.

    Notes
    -----
    Not implemented yet. The real implementation depends on the Aalto "136M
    Keystrokes" dataset (being downloaded/prepared separately) so that the
    HL/IL/PL/RL windowing logic can be built and validated against real data
    rather than guessed at. See CLAUDE.md's "Data" section for the full spec.
    """
    raise NotImplementedError(
        "feature extraction not yet implemented — see CLAUDE.md Data section"
    )
