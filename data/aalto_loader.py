"""Load Aalto "136M Keystrokes" participant files into per-sentence sessions.

Aalto rows are already press/release-paired (PRESS_TIME/RELEASE_TIME columns),
so this loader skips the browser-only event-pairing step in
`features.extract.pair_browser_events` and builds pairs directly, then hands
them to the same canonical `features.extract.windows_from_keystrokes` used by
live capture.
"""

from __future__ import annotations

import csv
from pathlib import Path


class Session:
    """One typed sentence: a single TEST_SECTION_ID within a participant file."""

    def __init__(self, participant_id: str, test_section_id: str, sentence: str):
        self.participant_id = participant_id
        self.test_section_id = test_section_id
        self.sentence = sentence
        self.pairs: list[tuple[float, float]] = []


def load_participant_file(path: Path) -> list[Session]:
    """Parse one participant's keystroke file into one Session per sentence typed.

    Uses `quoting=csv.QUOTE_NONE`: Aalto's TSV isn't quote-escaped, and some
    sentences contain literal ``"`` characters. Python's default CSV quoting
    treats those as the start of a quoted field and silently merges dozens of
    subsequent rows into one field until it finds a closing quote -- this
    corrupts data without raising an error, so QUOTE_NONE is required, not
    optional.
    """
    sessions: dict[str, Session] = {}

    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            try:
                press_t = float(row["PRESS_TIME"])
                release_t = float(row["RELEASE_TIME"])
            except (KeyError, ValueError, TypeError):
                continue
            if release_t < press_t:
                continue  # rare logging glitch, drop the single bad keystroke

            sid = row["TEST_SECTION_ID"]
            session = sessions.get(sid)
            if session is None:
                session = Session(
                    participant_id=row["PARTICIPANT_ID"],
                    test_section_id=sid,
                    sentence=row["SENTENCE"],
                )
                sessions[sid] = session
            session.pairs.append((press_t, release_t))

    for session in sessions.values():
        session.pairs.sort(key=lambda pair: pair[0])

    return list(sessions.values())


def iter_participant_files(raw_dir: Path):
    """Yield paths to all participant keystroke files in `raw_dir`."""
    yield from sorted(raw_dir.glob("*_keystrokes.txt"))
