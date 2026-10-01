"""TypeNet's subject split: the first 68,000 subjects (by numeric id) train, the rest are held out.

TypeNet IV-D: "we train the models using only the first 68,000 subjects from the Dhakal dataset";
Sec. V: "The remaining 100,000 subjects were employed only for model evaluation". Ids come from the raw
file names, not from a cache, so the split covers every subject whatever a cache later drops.

run with: python -m data_v2.make_split
"""

import json

from data.config import AALTO_RAW_PATH, DATA_ROOT
from data.aalto_loader import iter_participant_files

SPLIT_PATH = DATA_ROOT / "split_typenet.json"
TRAIN_SUBJECTS = 68_000


def main():
    subjects = sorted((p.name.removesuffix("_keystrokes.txt") for p in iter_participant_files(AALTO_RAW_PATH)), key=int)
    train, held_out = subjects[:TRAIN_SUBJECTS], subjects[TRAIN_SUBJECTS:]
    SPLIT_PATH.write_text(json.dumps({
        "rule": "first 68000 by numeric id (TypeNet IV-D)",
        "train_subjects": train,
        "eval_subjects": held_out,
    }))
    print(f"Saved {SPLIT_PATH}: {len(train)} train / {len(held_out)} eval subjects "
          f"(train ids {train[0]}..{train[-1]}, eval ids {held_out[0]}..{held_out[-1]})")


if __name__ == "__main__":
    main()
