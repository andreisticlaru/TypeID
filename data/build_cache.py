'''Time feature extraction across the Aalto dataset (no caching yet).

run with: python -m data.build_cache --limit 1000'''

import argparse
import logging
import time

from data.aalto_loader import load_participant_file, iter_participant_files
from data.config import AALTO_RAW_PATH
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

    start = time.perf_counter()

    # TODO: loop over iter_participant_files(AALTO_RAW_PATH), but stop once
    # you've processed args.limit files. You need an index to compare against
    # args.limit -- enumerate() gives you one for free.
    for i, path in enumerate(iter_participant_files(AALTO_RAW_PATH)):
        if participants_processed >= args.limit:
            break

        participants_processed += 1

        # TODO: load this participant's sessions
        sessions = load_participant_file(path)

        for session in sessions:
            # TODO: call windows_from_keystrokes(session.pairs).
            # Wrap it in try/except ValueError:
            #   - success -> sessions_ok += 1
            #   - ValueError (too short) -> sessions_skipped += 1
            try:
                windows_from_keystrokes(session.pairs)
                sessions_ok += 1
            except ValueError:
                sessions_skipped += 1

        if participants_processed % 1000 == 0:
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


if __name__ == "__main__":
    main()
