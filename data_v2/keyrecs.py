"""Cache KeyRecs (Zenodo 7886743, CC BY 4.0) free text for zero-shot scoring with comparison.paired.

KeyRecs gives digraphs, not raw events: per consecutive key pair, key1's hold (DU.key1.key1), press-to-press
(DD.key1.key2) and key1-press-to-key2-release (DU.key1.key2), in seconds. Chaining them rebuilds absolute
(press, release) times exactly, which then go through the canonical extractor like Aalto's pairs (sorted by
press, every key kept). A row that doesn't parse or doesn't chain (key2 != next key1) ends a run; no sequence
spans two runs.

Each person has 2 long sessions (~2,800 keys). The protocol wants 15 short sequences per person in time order,
so each is one 51-keystroke chunk (one full 50-step window), evenly spaced through the session:
  within: all 15 from session 1 (same sitting)
  cross:  10 from session 1 (gallery), 5 from session 2 (queries); Rank-1 and EER G are cross-session
  kvc_cross: 5 from session 1, 10 from session 2; only the KVC-style columns (5 enroll, 10 query) are
             cross-session, since comparison.paired's KVC metric enrolls the first 5 and queries the last 10

run with: python -m data_v2.keyrecs --mode within
"""

import argparse
import csv
import json

import numpy as np

from data.config import DATA_ROOT
from eval.typenet_protocol import GALLERY_MAX, SEQUENCES
from features.extract import M, windows_from_keystrokes

RAW_PATH = DATA_ROOT / "keyrecs_raw" / "free-text.csv"
CHUNK = M + 1  # 51 keystrokes -> 50 timing vectors


def parse(row: list[str]):
    try:
        return row[2], row[3], float(row[4]), float(row[5]), float(row[6])
    except (ValueError, IndexError):
        return None


def runs_to_pairs(digraphs) -> list[list[tuple[float, float]]]:
    """(press_ms, release_ms) pairs per unbroken run of parsed digraphs (None = break), each sorted by press."""
    runs, run = [], []
    for d in list(digraphs) + [None]:
        if run and (d is None or d[0] != run[-1][1]):
            runs.append(run)
            run = []
        if d is not None:
            run.append(d)
    out = []
    for run in runs:
        pairs, press = [], 0.0
        for _, _, hold, dd, du in run:
            pairs.append((press, press + 1000 * hold))
            last_release = press + 1000 * du  # key2's release, measured from key1's press
            press += 1000 * dd
        pairs.append((press, last_release))
        out.append(sorted(pairs))
    return out


def chunks(runs: list[list[tuple[float, float]]], n: int) -> list[list[tuple[float, float]]]:
    """`n` CHUNK-keystroke pieces, evenly spaced over all whole pieces of the runs, in time order."""
    pieces = [run[i:i + CHUNK] for run in runs for i in range(0, len(run) - CHUNK + 1, CHUNK)]
    if len(pieces) < n:
        return []
    return [pieces[i] for i in np.linspace(0, len(pieces) - 1, n).round().astype(int)]


def load_sessions() -> dict[str, dict[str, list]]:
    """{participant: {session: [parsed digraph or None]}} in file order."""
    people: dict[str, dict[str, list]] = {}
    with open(RAW_PATH, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            people.setdefault(row[0], {}).setdefault(row[1], []).append(parse(row))
    return people


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["within", "cross", "kvc_cross"], required=True)
    args = parser.parse_args()

    plan = {"within": [("1", SEQUENCES)], "cross": [("1", GALLERY_MAX), ("2", SEQUENCES - GALLERY_MAX)],
            "kvc_cross": [("1", 5), ("2", SEQUENCES - 5)]}[args.mode]
    windows, masks, subjects, sessions, dropped = [], [], [], [], []
    for person, by_session in sorted(load_sessions().items()):
        sequences = []
        for session, n in plan:
            got = chunks(runs_to_pairs(by_session.get(session, [])), n)
            sequences += got if len(got) == n else [None]
        if None in sequences:
            dropped.append(person)
            continue
        for i, pairs in enumerate(sequences, 1):
            w, m = windows_from_keystrokes(pairs)
            assert len(w) == 1 and m.all(), "a 51-keystroke chunk is one full window"
            windows.append(w)
            masks.append(m)
            subjects.append(person)
            sessions.append(str(i))  # time order, as typenet_protocol sorts session ids by int

    out = DATA_ROOT / f"preprocessed_keyrecs_{args.mode}"
    out.mkdir(exist_ok=True)
    np.save(out / "windows.npy", np.concatenate(windows))
    np.save(out / "mask.npy", np.concatenate(masks))
    np.save(out / "subject_ids.npy", np.array(subjects))
    np.save(out / "session_ids.npy", np.array(sessions))
    kept = sorted(set(subjects))
    (DATA_ROOT / f"split_keyrecs_{args.mode}.json").write_text(json.dumps({"rule": "KeyRecs free text, all eval (zero-shot)",
                                      "train_subjects": [], "eval_subjects": kept}))
    print(f"Saved {len(windows)} sequences for {len(kept)} people to {out}; dropped {dropped}")


if __name__ == "__main__":
    main()
