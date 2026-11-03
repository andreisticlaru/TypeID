"""The task gap: one encoder on the same people under different layouts of their typing (data_v2/klicke.py).

comparison.paired compares models on one cache; this compares caches for one model. Its scoring and bootstrap don't
care what a column is, so each layout takes a model's place: same people, same seeds and same resamples, which makes
every difference between layouts paired over people. The first --layout is the reference, and delta = layout - it.

Each metric is only cross-task in the layouts built for it, so the gap is read as:
  KVC EER / FNMR:   copy_essay - copy (same enrollment); copy_essay - essay (same queries)
  Rank-1, DIR, EER G: copy_essay_id - copy (same gallery)

run with: python -m comparison.task_gap --split data/split_klicke_dev.json --tag klicke_dev \
          --model C1=runs/p6_c_f/final.pt --model v1=model/encoder_hard.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from comparison.loaders import PauseEdit, load_any_encoder
from comparison.paired import RESULTS_DIR, load_people_embeddings, score_model, summarize
from data.config import DATA_ROOT
from data_v2.datasets import load_eval_data

LAYOUTS = ("copy", "copy_essay", "copy_essay_id", "essay")
COLUMNS = ("kvc_eer", "kvc_fnmr1", "rank1", "dir_far1", "eer_g1")


def cache_dir(dataset: str, layout: str) -> Path:
    return DATA_ROOT / f"preprocessed_{dataset}_{layout}"


def score_layouts(model, split, dataset, layouts, pauses, repetitions, background, device) -> tuple[dict, list[str]]:
    """Scores per layout, keyed "layout" for plain input and "layout+edit" for a PauseEdit of it."""
    scores, people = {}, None
    for layout in layouts:
        data = load_eval_data(split, cache_dir(dataset, layout))
        for pause in pauses:
            edited = model if pause == "none" else PauseEdit(model, pause)
            embeddings, these = load_people_embeddings(edited, data, device)
            assert people is None or these == people, "layouts must be scored on the same people"
            people = these
            key = layout if pause == "none" else f"{layout}+{pause}"
            scores[key] = score_model(embeddings, repetitions, background, device)
    return scores, people


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--split", required=True)
    parser.add_argument("--dataset", default="klicke")
    parser.add_argument("--layout", action="append", default=None, help=f"first is the reference; default: {LAYOUTS}")
    parser.add_argument("--pause", action="append", default=None, choices=["none", "cap", "soft", "noise"],
                        help="input edits to score each layout under (comparison.loaders.PauseEdit); default: none")
    parser.add_argument("--tag", required=True, help="result file is task_gap_<tag>.json")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--background", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    layouts = args.layout or LAYOUTS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"split": Path(args.split).as_posix(), "repetitions": args.repetitions, "background": args.background,
              "checkpoints": dict(s.split("=", 1) for s in args.model), "models": {}}
    for spec in args.model:
        name, path = spec.split("=", 1)
        scores, people = score_layouts(load_any_encoder(path, device), args.split, args.dataset, layouts,
                                       args.pause or ["none"], args.repetitions, args.background, device)
        row = summarize(scores, len(people), args.bootstrap)
        report["models"][name] = row
        report["people"] = len(people)
        print(f"\n{name}: {len(people)} people [95% CI over people]", flush=True)
        print(f"{'layout':<22}" + "".join(f"{c:>24}" for c in COLUMNS))
        for layout, r in row["models"].items():
            print(f"{layout:<22}" + "".join(f"{r[c][0]:>8.2f} [{r[c][1]:.2f},{r[c][2]:.2f}]".rjust(24) for c in COLUMNS))
        for layout, r in row["models"].items():
            if "vs_reference" in r:
                print(f"{layout + ' - ' + row['reference']:<30}" + "  ".join(
                    f"{c} {r['vs_reference'][c]['delta']:+.2f} [{r['vs_reference'][c]['ci95'][0]:+.2f},{r['vs_reference'][c]['ci95'][1]:+.2f}]"
                    for c in COLUMNS), flush=True)
        pair = row["pairwise"]["kvc_eer"]
        for a, b in [("copy_essay", "essay")] + [(f"{l}+{p}", l) for l in layouts for p in (args.pause or [])]:
            if a in pair and b in pair[a]:
                d = pair[a][b]
                print(f"{a} - {b}  kvc_eer {d['delta']:+.2f} [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}]", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    json.dump(report, open(RESULTS_DIR / f"task_gap_{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
