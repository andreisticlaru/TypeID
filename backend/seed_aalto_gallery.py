"""Fill the gallery with held-out Aalto people so /identify has real distractors.

Each person is enrolled exactly as /enroll does it: embed each of 5 sessions, then mean-pool. Only
eval subjects (never seen in training) are used, from the cached windows, so the features are the
ones the canonical extractor already produced.

run from backend/:  ../.venv-model/Scripts/python seed_aalto_gallery.py --count 500
undo:               ../.venv-model/Scripts/python seed_aalto_gallery.py --remove
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.embedding import embed
from app.gallery import DB_PATH, add_entry, init_db

PREFIX = "aalto_"
ENROLL_SESSIONS = 5  # same as the frontend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()

    init_db()
    if args.remove:
        with sqlite3.connect(DB_PATH) as conn:
            removed = conn.execute("DELETE FROM gallery WHERE person_id LIKE ?", (PREFIX + "%",)).rowcount
        print(f"removed {removed} Aalto entries")
        return

    pre = REPO_ROOT / "data" / "preprocessed"
    windows = np.load(pre / "windows.npy", mmap_mode="r")
    mask = np.load(pre / "mask.npy", mmap_mode="r")
    subject_ids = np.load(pre / "subject_ids.npy")
    session_ids = np.load(pre / "session_ids.npy")
    eval_subjects = json.load(open(REPO_ROOT / "data" / "split.json"))["eval_subjects"]

    rows = np.flatnonzero(np.isin(subject_ids, eval_subjects))
    sessions: dict[str, dict[str, list[int]]] = {}
    for r in rows:
        sessions.setdefault(subject_ids[r], {}).setdefault(session_ids[r], []).append(r)
    eligible = sorted(s for s, d in sessions.items() if len(d) >= ENROLL_SESSIONS)
    chosen = np.random.default_rng(args.seed).choice(eligible, size=min(args.count, len(eligible)), replace=False)

    for subject in chosen:
        by_session = sessions[subject]
        first_sessions = sorted(by_session, key=int)[:ENROLL_SESSIONS]  # first 5 in time order
        embeddings = [embed((np.array(windows[by_session[s]]), np.array(mask[by_session[s]]))) for s in first_sessions]
        add_entry(PREFIX + subject, PREFIX + subject, np.mean(np.stack(embeddings), axis=0).tolist())
    print(f"enrolled {len(chosen)} held-out Aalto people (of {len(eligible)} eligible) into {DB_PATH}")


if __name__ == "__main__":
    main()
