"""Cache the Aalto sessions under TypeNet's sequence rule: one sequence per session, first 50 steps, no floor.

TypeNet IV-B: "truncate the end of the input sequence when N>M and zero pad at the end when N<M". So each
session gives exactly one window, from its first 51 keystrokes (N keystrokes -> N-1 timing vectors), however
short it is. Sessions with fewer than 2 keystrokes have no timing vector at all and are skipped.

Same four arrays as data/build_cache.py, so every loader works on either cache. The features come from the
canonical extractor; only its floor is lowered.

run with: python -m data_v2.build_cache          (full dataset, a few minutes on 16 cores)
          python -m data_v2.build_cache --limit 200 --out <scratch dir>
"""

import argparse
import logging
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data.aalto_loader import iter_participant_files, load_participant_file
from data.config import AALTO_RAW_PATH, DATA_ROOT, PREPROCESSED_PATH
from features.extract import M, windows_from_keystrokes

CACHE_PATH = DATA_ROOT / "preprocessed_typenet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def participant_windows(path: Path):
    """(windows, mask, subject_ids, session_ids, skipped) for one participant file."""
    windows, masks, subjects, sessions, skipped = [], [], [], [], 0
    for session in load_participant_file(path):
        try:
            w, m = windows_from_keystrokes(session.pairs[: M + 1], min_keystrokes=1)
        except ValueError:
            skipped += 1
            continue
        windows.append(w)
        masks.append(m)
        subjects.append(session.participant_id)
        sessions.append(session.test_section_id)
    return windows, masks, subjects, sessions, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="max participant files (default: all)")
    parser.add_argument("--out", type=Path, default=CACHE_PATH)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    out = args.out.resolve()
    if out == PREPROCESSED_PATH.resolve() or PREPROCESSED_PATH.resolve() in out.parents:
        raise SystemExit(f"refusing to write into v1's cache {PREPROCESSED_PATH}")

    files = list(iter_participant_files(AALTO_RAW_PATH))[: args.limit]
    all_windows, all_masks, all_subjects, all_sessions, skipped = [], [], [], [], 0
    start = time.perf_counter()
    with Pool(args.workers) as pool:
        for i, (w, m, subj, sess, s) in enumerate(pool.imap(participant_windows, files, chunksize=64), 1):
            all_windows += w
            all_masks += m
            all_subjects += subj
            all_sessions += sess
            skipped += s
            if i % 5000 == 0:
                logger.info(f"{i}/{len(files)} participants, {len(all_windows)} sessions kept, {skipped} skipped, "
                            f"{time.perf_counter() - start:.0f}s")

    windows = np.concatenate(all_windows)
    mask = np.concatenate(all_masks)
    assert len(windows) == len(all_subjects), "one window per session"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "windows.npy", windows)
    np.save(out / "mask.npy", mask)
    np.save(out / "subject_ids.npy", np.array(all_subjects))
    np.save(out / "session_ids.npy", np.array(all_sessions))
    lengths = mask.sum(axis=1)
    logger.info(f"Saved {len(windows)} windows from {len(files)} participants to {out}; {skipped} sessions skipped "
                f"(<2 keystrokes); {100 * (lengths == M).mean():.1f}% full {M}-step; median length "
                f"{int(np.median(lengths))}; {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
