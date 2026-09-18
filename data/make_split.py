"""Create the subject-disjoint train/eval split and save it to data/split.json.

Split by subject, never by sample or session: eval must measure generalization
to people the model has never seen. The file is saved (not just re-derived from
a seed) so training and eval always use the identical held-out subjects, even
if the cache is rebuilt.

run with: python -m data.make_split
"""

import argparse
import json

import numpy as np

from data.config import DATA_ROOT, PREPROCESSED_PATH

SPLIT_PATH = DATA_ROOT / "split.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subjects = np.unique(np.load(PREPROCESSED_PATH / "subject_ids.npy"))
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(subjects)

    n_eval = int(len(shuffled) * args.eval_fraction)
    eval_subjects = sorted(shuffled[:n_eval].tolist())
    train_subjects = sorted(shuffled[n_eval:].tolist())

    SPLIT_PATH.write_text(
        json.dumps({"seed": args.seed, "train_subjects": train_subjects, "eval_subjects": eval_subjects})
    )
    print(f"Saved {SPLIT_PATH}: {len(train_subjects)} train / {len(eval_subjects)} eval subjects")


if __name__ == "__main__":
    main()
