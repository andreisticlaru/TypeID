'''Extract timing features across the Aalto dataset and cache them to disk.

run with: python -m data.build_cache --limit 1000'''

import argparse
import logging
import time

import numpy as np

from data.aalto_loader import load_participant_file, iter_participant_files
from data.config import AALTO_RAW_PATH, PREPROCESSED_PATH
from features.extract import windows_from_keystrokes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000, help="max participant files to process")
    args = parser.parse_args()

    sessions_ok = 0
    sessions_skipped = 0
    participants_processed = 0

    all_windows = []
    all_masks = []
    all_subject_ids = []
    all_session_ids = []

    start = time.perf_counter()

    for i, path in enumerate(iter_participant_files(AALTO_RAW_PATH)):
        if participants_processed >= args.limit:
            break

        participants_processed += 1
        sessions = load_participant_file(path)

        for session in sessions:
            try:
                windows, mask = windows_from_keystrokes(session.pairs)
                sessions_ok += 1
            except ValueError:
                sessions_skipped += 1
                continue

            all_windows.append(windows)
            all_masks.append(mask)
            all_subject_ids.extend([session.participant_id] * len(windows))
            all_session_ids.extend([session.test_section_id] * len(windows))

        if participants_processed % 100 == 0:
            logger.info(
                f"Processed {participants_processed} participants, sessions_ok={sessions_ok}, "
                f"sessions_skipped={sessions_skipped}, elapsed={time.perf_counter() - start:.2f}s"
            )

    elapsed = time.perf_counter() - start

    total_sessions = sessions_ok + sessions_skipped
    seconds_per_session = elapsed / total_sessions if total_sessions > 0 else 0

    logger.info(
        f"Final summary: Processed {participants_processed} participants, "
        f"sessions_ok={sessions_ok}, sessions_skipped={sessions_skipped}, "
        f"elapsed={elapsed:.2f}s, "
        f"seconds_per_session={seconds_per_session:.4f}s, "
        f"extrapolated_full_dataset_time={seconds_per_session * 2_400_000:.2f}s"
    )

    windows_arr = np.concatenate(all_windows, axis=0)
    mask_arr = np.concatenate(all_masks, axis=0)
    subject_ids_arr = np.array(all_subject_ids)
    session_ids_arr = np.array(all_session_ids)

    np.save(PREPROCESSED_PATH / "windows.npy", windows_arr)
    np.save(PREPROCESSED_PATH / "mask.npy", mask_arr)
    np.save(PREPROCESSED_PATH / "subject_ids.npy", subject_ids_arr)
    np.save(PREPROCESSED_PATH / "session_ids.npy", session_ids_arr)

    logger.info(
        f"Saved {windows_arr.shape[0]} windows to {PREPROCESSED_PATH} "
        f"(windows.npy shape={windows_arr.shape})"
    )


if __name__ == "__main__":
    main()
