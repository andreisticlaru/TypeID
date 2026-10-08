"""The fair test set for v1 vs the new models: people held out by BOTH v1's split and TypeNet's split.

v1 (model/encoder_hard.pt) trained on a random 90% of everyone (data/split.json), so it has seen most of TypeNet's
test people; the new models trained on the first 68,000 ids, which include ~40% of v1's test people. Neither has
seen anyone in the intersection. train_subjects is empty: the file is for comparison.evaluate --split only.

run with: python -m data_v2.make_unseen_split
"""

import json

from data.config import DATA_ROOT
from data_v2.datasets import load_split
from data_v2.make_split import SPLIT_PATH as TYPENET_SPLIT_PATH

V1_SPLIT_PATH = DATA_ROOT / "split.json"
UNSEEN_SPLIT_PATH = DATA_ROOT / "split_unseen_both.json"


def main():
    v1_eval = set(load_split(V1_SPLIT_PATH)["eval_subjects"])
    unseen = sorted(v1_eval & set(load_split(TYPENET_SPLIT_PATH)["eval_subjects"]), key=int)
    UNSEEN_SPLIT_PATH.write_text(json.dumps({
        "rule": "v1 eval (data/split.json) ∩ TypeNet eval (data/split_typenet.json)",
        "train_subjects": [],
        "eval_subjects": unseen,
    }))
    print(f"Saved {UNSEEN_SPLIT_PATH}: {len(unseen)} eval subjects")


if __name__ == "__main__":
    main()
