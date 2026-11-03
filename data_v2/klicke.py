"""Cache KLiCKe (Inputlog logs of ~5,000 writers: copy tasks + a ~30 min essay) for zero-shot scoring.

Read straight from the corpus's two zips: they are complete and need no 10 GB extraction. Only the csv logs' times
and event names are used, never the text.

What counts as a keystroke (the rest would not exist in Aalto or the live app, which only see real key timings):
  * mouse events are dropped; the time they took stays in the next interval, as in a browser capture.
  * zero-hold events are dropped. The logger never times modifiers: Shift and Control are 0 ms in 100% of rows.
    The copy tasks are all lowercase, so keeping these would add a fake hold time to essays only, a task gap made
    by the logger.

Copy text is TYPING2 and TYPING3_1-4: one short phrase retyped at speed, not Aalto-style varied sentences. TYPING1
(tapping "dkdk") and TYPING4 (27 random consonants) are not text. A sequence is one 51-keystroke chunk (one full
window), and no chunk spans two tasks.

Layouts share their sequences, so each comparison changes one thing:
  copy:       first 15 copy chunks
  copy_essay: first 5 copy (same enrollment as copy) + 10 from the essay's second half; KVC columns are cross-task
  copy_essay_id: first 10 copy (same gallery as copy) + last 5 of those 10; Rank-1 and EER G are cross-task
  essay:      5 from the essay's first half + the same 10 as copy_essay; enrollment differs from copy_essay only
  copy_burst: first 5 copy + 10 second-half essay chunks with no interval above Aalto's 99th percentile, so the
              queries are composed text with its long pauses absent, not just shortened (split_klicke_dev_burst.json)

The split is by person, over everyone, before eligibility: a dev half for choosing fixes and a test half to score
once. split_klicke_{dev,test}.json hold people who fit every layout, so layouts compare the same people. Only 54%
have 15 copy chunks and they type faster, so split_klicke_dev_wide.json has every dev person who fits copy_essay,
to check that this selection doesn't drive the result.

run with: python -m data_v2.klicke
"""

import csv
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np

from comparison.loaders import AALTO_IL_P99
from data.config import DATA_ROOT
from data_v2.keyrecs import CHUNK, chunks
from features.extract import windows_from_keystrokes

RAW_PATH = Path(r"C:\Users\andre\Downloads\KLiCKe corpus-20261102T020438Z-1-001\KLiCKe corpus")
COPY_TASKS = ("TYPING2", "TYPING3_1", "TYPING3_2", "TYPING3_3", "TYPING3_4")
LAYOUTS = ("copy", "copy_essay", "copy_essay_id", "essay")
BURST = "copy_burst"
SPLIT_SEED = 0


def log_to_pairs(text: str) -> list[tuple[float, float]]:
    """(press_ms, release_ms) per real keystroke of one Inputlog csv, sorted by press."""
    pairs = []
    reader = csv.reader(io.StringIO(text))
    next(reader, None)
    for row in reader:
        try:
            down, up, event = float(row[3]), float(row[4]), row[6]
        except (ValueError, IndexError):  # a literal quote key can break a row's csv quoting
            continue
        if up > down and not event.endswith("click"):
            pairs.append((down, up))
    return sorted(pairs)


def first_chunks(runs: list[list[tuple[float, float]]], n: int) -> list[list[tuple[float, float]]]:
    """The first `n` whole CHUNK-keystroke pieces of the runs, in time order; [] if there are fewer."""
    pieces = [run[i:i + CHUNK] for run in runs for i in range(0, len(run) - CHUNK + 1, CHUNK)]
    return pieces[:n] if len(pieces) >= n else []


def pause_free_pieces(run: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """Non-overlapping CHUNK-keystroke pieces of a run with no IL above AALTO_IL_P99, in time order."""
    pieces, i = [], 0
    while i + CHUNK <= len(run):
        piece = run[i:i + CHUNK]
        long_gaps = [j for j in range(CHUNK - 1) if piece[j + 1][0] - piece[j][1] > AALTO_IL_P99]
        if long_gaps:
            i += long_gaps[-1] + 1
        else:
            pieces.append(piece)
            i += CHUNK
    return pieces


def layouts_for(copy_runs: list[list[tuple[float, float]]], essay: list[tuple[float, float]]) -> dict[str, list]:
    """{layout: 15 chunks in protocol order} for the layouts this person has enough typing for."""
    half = len(essay) // 2
    early, late = chunks([essay[:half]], 5), chunks([essay[half:]], 10)
    copy5, copy10, copy15 = (first_chunks(copy_runs, n) for n in (5, 10, 15))
    out = {}
    if copy15:
        out["copy"] = copy15
    if copy5 and late:
        out["copy_essay"] = copy5 + late
    if copy10 and late:
        out["copy_essay_id"] = copy10 + late[5:]
    if early and late:
        out["essay"] = early + late
    bursts = chunks(pause_free_pieces(essay[half:]), 10)
    if copy5 and bursts:
        out[BURST] = copy5 + bursts
    return out


def split_people(people: list[str]) -> tuple[set[str], set[str]]:
    order = np.random.default_rng(SPLIT_SEED).permutation(sorted(people))
    half = len(order) // 2
    return set(order[:half].tolist()), set(order[half:].tolist())


def load_people(raw: Path):
    """Yield (person, copy_runs, essay_pairs) for everyone with an essay log."""
    with zipfile.ZipFile(raw / "TypingTests.zip") as typing, zipfile.ZipFile(raw / "WritingTask.zip") as writing:
        typing_names = set(typing.namelist())
        for name in sorted(writing.namelist()):
            found = re.fullmatch(r"WritingTask/keystrokelogs/csv/(\d+)\.csv", name)
            if not found:
                continue
            person = found[1]
            logs = [f"TypingTests/csv/{person}_{task}.csv" for task in COPY_TASKS]
            copy_runs = [log_to_pairs(typing.read(log).decode("utf-8", "replace")) for log in logs if log in typing_names]
            yield person, copy_runs, log_to_pairs(writing.read(name).decode("utf-8", "replace"))


def main():
    arrays = {layout: {"windows": [], "mask": [], "subject_ids": [], "session_ids": []} for layout in (*LAYOUTS, BURST)}
    fits, everyone = {}, []
    for i, (person, copy_runs, essay) in enumerate(load_people(RAW_PATH), 1):
        everyone.append(person)
        built = layouts_for(copy_runs, essay)
        fits[person] = set(built)
        for layout, sequences in built.items():
            for s, pairs in enumerate(sequences, 1):
                w, m = windows_from_keystrokes(pairs)
                assert len(w) == 1 and m.all(), "a 51-keystroke chunk is one full window"
                arrays[layout]["windows"].append(w)
                arrays[layout]["mask"].append(m)
                arrays[layout]["subject_ids"].append(person)
                arrays[layout]["session_ids"].append(str(s))  # time order, as typenet_protocol sorts ids by int
        if i % 500 == 0:
            print(f"{i} people", flush=True)

    for layout, a in arrays.items():
        out = DATA_ROOT / f"preprocessed_klicke_{layout}"
        out.mkdir(exist_ok=True)
        np.save(out / "windows.npy", np.concatenate(a["windows"]))
        np.save(out / "mask.npy", np.concatenate(a["mask"]))
        np.save(out / "subject_ids.npy", np.array(a["subject_ids"]))
        np.save(out / "session_ids.npy", np.array(a["session_ids"]))
        print(f"{layout}: {len(a['windows']) // 15} people", flush=True)

    dev, test = split_people(everyone)
    every_layout = {p for p, f in fits.items() if set(LAYOUTS) <= f}
    wide = {p for p, f in fits.items() if {"copy_essay", "essay"} <= f}
    splits = {"dev": dev & every_layout, "test": test & every_layout, "dev_wide": dev & wide,
              "dev_burst": dev & every_layout & {p for p, f in fits.items() if BURST in f}}
    for name, people in splits.items():
        (DATA_ROOT / f"split_klicke_{name}.json").write_text(json.dumps(
            {"rule": f"KLiCKe {name}, all eval (zero-shot), split by person", "train_subjects": [],
             "eval_subjects": sorted(people)}))
        print(f"split {name}: {len(people)} people", flush=True)


if __name__ == "__main__":
    main()
