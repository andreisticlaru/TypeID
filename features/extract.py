"""Canonical feature extraction: raw keystroke events -> timing-feature sequence.

This module is the ONE implementation of feature extraction referenced throughout
CLAUDE.md ("Route live browser events to a backend endpoint that runs the one
canonical Python feature-extraction function"). It must be imported by both the
training pipeline (operating on rows from the Aalto dataset) and the backend
/enroll and /identify endpoints (operating on live browser-captured events) --
never duplicated or reimplemented in JS.

The Aalto dataset and live browser capture converge on the same representation
-- a time-ordered list of (press_time_ms, release_time_ms) pairs, one per
keystroke -- before either path reaches `windows_from_keystrokes`, which is the
single place the HL/IL/PL/RL math and M=50 windowing/padding/masking live.
Aalto rows already come pre-paired (PRESS_TIME/RELEASE_TIME columns); browser
keydown/keyup events need `pair_browser_events` first.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

M = 50  # fixed window length (TypeNet's value; keep identical training <-> live)
MIN_KEYSTROKES = 25  # below this, timing signal is too thin to trust (see CLAUDE.md)


def pair_browser_events(events: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Match keydown events to their keyup by key, return press/release pairs.

    Browser event streams can interleave across keys (e.g. Shift held down
    while W is pressed and released), so pairing can't assume the events for
    one key arrive back-to-back -- it has to track open presses per key and
    close the oldest open press for that key on the matching keyup.

    Parameters
    ----------
    events:
        Raw events shaped like ``{"key": str, "code": str, "type": "keydown"
        | "keyup", "t": float}`` with ``t`` a millisecond timestamp, exactly
        as captured by the frontend's keystroke logger.

    Returns
    -------
    list[tuple[float, float]]
        ``(press_time, release_time)`` pairs, sorted ascending by press time.
    """
    open_presses: dict[str, list[float]] = {}
    pairs: list[tuple[float, float]] = []

    for ev in events:
        key = ev.get("code", ev.get("key"))
        if ev["type"] == "keydown":
            open_presses.setdefault(key, []).append(ev["t"])
        elif ev["type"] == "keyup":
            pending = open_presses.get(key)
            if pending:
                press_t = pending.pop(0)
                pairs.append((press_t, ev["t"]))
            # keyup with no matching open keydown (e.g. mid-capture start) is dropped

    pairs.sort(key=lambda pair: pair[0])
    return pairs


def windows_from_keystrokes(
    pairs: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute HL/IL/PL/RL timing features and window them to fixed length M.

    This is the single canonical implementation of the timing math -- both
    the Aalto training pipeline and live `/enroll`/`/identify` requests must
    call this exact function on their respective (press, release) pairs.
    Never reimplement this arithmetic elsewhere.

    Parameters
    ----------
    pairs:
        ``(press_time_ms, release_time_ms)`` tuples, one per keystroke,
        sorted ascending by press time.

    Returns
    -------
    windows : np.ndarray, shape (num_windows, M, 4)
        Per-keystroke ``(HL, IL, PL, RL)`` vectors, zero-padded per window.
    mask : np.ndarray, shape (num_windows, M), dtype=bool
        True where `windows` holds real data, False where it's padding.

    Notes
    -----
    HL(n) only depends on keystroke n, but IL/PL/RL(n) depend on keystroke
    n+1 as well, so N keystrokes produce N-1 complete feature vectors (the
    last keystroke has no "next" to measure IL/PL/RL against). This follows
    standard practice in the keystroke-dynamics literature rather than
    inventing a value for an undefined boundary case.
    """
    if len(pairs) < MIN_KEYSTROKES + 1:
        raise ValueError(
            f"need at least {MIN_KEYSTROKES + 1} keystrokes to extract "
            f"{MIN_KEYSTROKES} timing vectors, got {len(pairs)}"
        )

    press = np.array([p for p, _ in pairs], dtype=np.float64)
    release = np.array([r for _, r in pairs], dtype=np.float64)

    hl = release[:-1] - press[:-1]
    il = press[1:] - release[:-1]
    pl = press[1:] - press[:-1]
    rl = release[1:] - release[:-1]
    features = np.stack([hl, il, pl, rl], axis=1).astype(np.float32)  # (N-1, 4)

    n = features.shape[0]
    num_windows = math.ceil(n / M)
    windows = np.zeros((num_windows, M, 4), dtype=np.float32)
    mask = np.zeros((num_windows, M), dtype=bool)

    for i in range(num_windows):
        start, end = i * M, min((i + 1) * M, n)
        length = end - start
        windows[i, :length] = features[start:end]
        mask[i, :length] = True

    return windows, mask


def extract_features(events: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw browser keydown/keyup events into windowed timing features.

    This is the entry point the backend's `/enroll` and `/identify` endpoints
    call. It's a thin wrapper: pair the raw events, then run the exact same
    `windows_from_keystrokes` used by the Aalto training pipeline.
    """
    return windows_from_keystrokes(pair_browser_events(events))
