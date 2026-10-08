"""Evaluate checkpoints of any architecture under both protocols and write one JSON per checkpoint.

  1. TypeNet's protocol (eval/typenet_protocol.py): 1,000 people, 10 gallery + 5 query sequences,
     one window per sequence, mean pairwise EUCLIDEAN distance on raw embeddings -- TypeNet's own rule,
     so its model is scored the way it was trained. Rank-1/5/20 and per-person EER at G = 1/2/5/7/10.
  2. Strict Rank-N (eval/rank_n.py): 1,000 people, 3 enrollment sessions + 1 query session, every window
     pooled, COSINE on the averaged profile -- the backend's scoring rule, so this is the number that
     predicts the live demo. It is not TypeNet's metric; TypeNet's raw embeddings were never trained for it.

run with: python -m comparison.evaluate --checkpoint runs/pilot_old_split/typenet_a.pt --seeds 0 1 2 3 4 5 6 7 8 9
TypeNet's split, no-floor cache:
          python -m comparison.evaluate --split data/split_typenet.json --cache data/preprocessed_typenet --checkpoint runs/p1_m0_nf/final.pt
Defaults are v1's split and cache, so earlier results reproduce. A checkpoint from typenet.train is saved as
results/<run_id>_<file stem>.json (its files are all called final.pt / step<N>.pt). --tag appends _<tag>, so
scoring the same checkpoint on another test set (e.g. data/split_unseen_both.json) never overwrites a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from comparison.loaders import load_any_encoder
from data.config import DATA_ROOT, PREPROCESSED_PATH
from data_v2.datasets import load_eval_data
from eval.rank_n import RANKS_REPORTED, evaluate_model, summarize
from eval.typenet_protocol import ENROLLMENT_SIZES, authentication_eer, draw_background, identification_ranks, session_embeddings

RESULTS_DIR = Path(__file__).parent / "results"


def mean_std(values) -> list[float]:
    values = np.asarray(values, dtype=float)
    return [round(float(values.mean()), 3), round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 3)]


def typenet_protocol(model, data, device, seeds, background_size=1000) -> dict:
    embeddings = session_embeddings(model, data, device)
    ranks, eers = [], {g: [] for g in ENROLLMENT_SIZES}
    for seed in seeds:
        background = draw_background(embeddings, seed, background_size)
        ranks.append(identification_ranks(background).cpu().numpy())
        rng = np.random.default_rng(1000 + seed)
        for g in ENROLLMENT_SIZES:
            eers[g].append(authentication_eer(background, g, rng))
    return {
        "metric": "mean pairwise Euclidean distance, raw embeddings",
        "eligible_people": len(embeddings),
        **{f"rank{n}": mean_std([100 * (r <= n).mean() for r in ranks]) for n in (1, 5, 20, 50)},
        **{f"eer_g{g}": mean_std(v) for g, v in eers.items()},
    }


def strict_rank_n(model, data, device, seeds) -> dict:
    mean, std = summarize(evaluate_model(model, data, device, seeds, num_subjects=1000, enroll_sessions=3, score="mean"))
    return {
        "metric": "cosine on averaged profile (backend rule), 3 enroll + 1 query",
        **{f"rank{n}": [round(100 * float(m), 3), round(100 * float(s), 3)] for n, m, s in zip(RANKS_REPORTED, mean, std)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--split", default=str(DATA_ROOT / "split.json"))
    parser.add_argument("--cache", default=str(PREPROCESSED_PATH))
    parser.add_argument("--tag", default=None, help="suffix for the result file name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_eval_data(args.split, args.cache)
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"{len(args.seeds)} seeds. TypeNet protocol: Rank-1 (Rank-50) | EER% G=1/5/10.  Strict: Rank-1/5/10/20")
    for path in map(Path, args.checkpoint):
        config = torch.load(path, map_location="cpu")["config"]
        model = load_any_encoder(path, device)
        result = {
            "checkpoint": str(path).replace("\\", "/"),
            "arch": config.get("arch", "v1"),
            "steps": config["steps"],
            "batch_size": config["batch_size"],
            "triplets_seen": config["steps"] * config["batch_size"],
            "seeds": args.seeds,
            "eval_split": Path(args.split).as_posix(),
            "eval_cache": Path(args.cache).as_posix(),
            "train_cache": Path(config["cache"]).as_posix() if "cache" in config else "data/preprocessed",
            "typenet_protocol": typenet_protocol(model, data, device, args.seeds),
            "strict": strict_rank_n(model, data, device, args.seeds),
        }
        name = f"{config['run_id']}_{path.stem}" if "run_id" in config else path.stem
        name = f"{name}_{args.tag}" if args.tag else name
        json.dump(result, open(RESULTS_DIR / f"{name}.json", "w"), indent=2)
        t, s = result["typenet_protocol"], result["strict"]
        print(f"{name:<34} {t['rank1'][0]:5.1f} (R50 {t['rank50'][0]:5.1f}) | {t['eer_g1'][0]:.2f}/{t['eer_g5'][0]:.2f}/{t['eer_g10'][0]:.2f}"
              f"   strict {s['rank1'][0]:5.1f}/{s['rank5'][0]:5.1f}/{s['rank10'][0]:5.1f}/{s['rank20'][0]:5.1f}   eligible {t['eligible_people']}", flush=True)


if __name__ == "__main__":
    main()
