"""KVC-scale split: TypeNet's 68,000 train subjects plus 47,120 more = 115,120, KVC's desktop development size.

The extra subjects come from TypeNet's held-out people minus the 10,001 in split_unseen_both.json, so the
training set contains C1's (only the number of people changes) and the unseen_both test people stay unseen.
Eval = those same 10,001. A model trained on this split can't be scored on TypeNet's paper view.

run with: python -m data_v2.make_scale_split
"""

import json

import numpy as np

from data.config import DATA_ROOT
from data_v2.make_split import SPLIT_PATH
from data_v2.make_unseen_split import UNSEEN_SPLIT_PATH

SCALE_SPLIT_PATH = DATA_ROOT / "split_kvc_scale.json"
TRAIN_SUBJECTS = 115_120
SEED = 0


def main():
    typenet = json.loads(SPLIT_PATH.read_text())
    unseen = json.loads(UNSEEN_SPLIT_PATH.read_text())["eval_subjects"]
    pool = sorted(set(typenet["eval_subjects"]) - set(unseen), key=int)
    extra = np.random.default_rng(SEED).choice(pool, TRAIN_SUBJECTS - len(typenet["train_subjects"]), replace=False)
    train = typenet["train_subjects"] + sorted(extra.tolist(), key=int)
    SCALE_SPLIT_PATH.write_text(json.dumps({
        "rule": f"TypeNet train + {len(extra)} random (seed {SEED}) from TypeNet eval minus unseen_both",
        "train_subjects": train,
        "eval_subjects": unseen,
    }))
    print(f"Saved {SCALE_SPLIT_PATH}: {len(train)} train / {len(unseen)} eval subjects")


if __name__ == "__main__":
    main()
