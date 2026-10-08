"""Train one model of the controlled study: TypeNet with ablation flags, or v1's encoder, on any split and cache.

  M0 (TypeNet, lr 1e-3):   python -u -m typenet.train --run-id p1_m0_nf
  M1 (+ L2, cosine):        ... --run-id p1_m1_nf --normalize --distance cosine --margin 0.5
  M4 (+ semi-hard):         ... --run-id p1_m4_nf --semi-from 10000
  M7 (v1 recipe):           ... --run-id p1_m7_nf --arch v1 --batch-size 64 --steps 240000 --semi-from 96000 --hard-from 144000

Writes runs/<run_id>/log.txt (with -u and a redirect), step<N>.pt every --save-every steps and final.pt.
Defaults: TypeNet's split and no-floor cache, batch 512, 30,000 steps (the paper's 15.36M triplets).

Paper recipe (Sec 4.3): Adam lr 0.05, betas (0.9, 0.999), eps 1e-8. lr 0.05 saturates this implementation
(pilot: 89% of second-layer units constant, Rank-1 1%), so every run here uses lr 1e-3.

Negatives: random until --semi-from, semi-hard until --hard-from, hardest after (a flag left out = never).
A collapse under hardest mining (all embeddings on one point, loss pinned at the margin) is a result, not a
crash: the guard saves a checkpoint, logs COLLAPSED and exits 0.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from data_v2.build_cache import CACHE_PATH
from data_v2.datasets import make_sampler
from data_v2.make_split import SPLIT_PATH
from model.network import KeystrokeEncoder
from model.train import load_for_resume, save_checkpoint, to_tensors
from typenet.losses import MARGIN, mined_loss, triplet_loss
from typenet.network import TypeNetEncoder

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
BATCH_SIZE = 512
STEPS = 30_000
LR = 1e-3
COSINE_MARGIN = 0.5  # model/losses.py
LOG_EVERY = 100
COLLAPSE_STEPS = 2_000


def negatives_mode(step: int, semi_from: int | None, hard_from: int | None) -> str:
    if hard_from is not None and step > hard_from:
        return "hard"
    if semi_from is not None and step > semi_from:
        return "semi"
    return "random"


def embed_triplets(model, batch, arch: str):
    if arch == "v1":  # no BatchNorm, so three calls equal one grouped call
        return (model(batch["anchor_windows"], batch["anchor_masks"]),
                model(batch["positive_windows"], batch["positive_masks"]),
                model(batch["negative_windows"], batch["negative_masks"]))
    # One pass over anchor+positive+negative; groups=3 keeps BatchNorm statistics per branch, as a Keras
    # siamese model computes them (tested equal to three separate calls).
    x = torch.cat([batch["anchor_windows"], batch["positive_windows"], batch["negative_windows"]])
    mask = torch.cat([batch["anchor_masks"], batch["positive_masks"], batch["negative_masks"]])
    return model(x, mask, groups=3).chunk(3)


def train_step(model, optimizer, batch, arch, mode, margin, distance) -> tuple[float, float, float]:
    """Returns (loss, mean embedding norm, mean pairwise Euclidean distance between anchors).

    The last two are the collapse detectors: under a collapse the spread falls towards 0."""
    anchor, positive, negative = embed_triplets(model, batch, arch)
    if mode == "random":
        loss = triplet_loss(anchor, positive, negative, margin, distance)
    else:
        pool_subjects = np.concatenate([batch["anchor_subjects"], batch["negative_subjects"]])
        same_subject = torch.from_numpy(batch["anchor_subjects"][:, None] == pool_subjects[None, :]).to(anchor.device)
        loss = mined_loss(anchor, positive, negative, same_subject, mode, margin, distance)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    a = anchor.detach()
    return loss.item(), a.norm(dim=1).mean().item(), torch.pdist(a).mean().item()


def build_model(args) -> tuple[torch.nn.Module, dict]:
    if args.arch == "v1":
        return KeystrokeEncoder(), {}
    kwargs = {"normalize": args.normalize, "readout": args.readout, "batchnorm": not args.no_batchnorm}
    return TypeNetEncoder(**kwargs), kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arch", choices=["typenet", "v1"], default="typenet")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize TypeNet's output")
    parser.add_argument("--readout", choices=["last", "mean"], default="last")
    parser.add_argument("--no-batchnorm", action="store_true")
    parser.add_argument("--distance", choices=["sqeuclidean", "cosine"], default=None,
                        help="default: sqeuclidean for typenet, cosine for v1 (forced)")
    parser.add_argument("--margin", type=float, default=None, help="default: 1.5 sqeuclidean, 0.5 cosine")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=STEPS, help="total steps (with --resume, the target, not extra)")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--semi-from", type=int, default=None)
    parser.add_argument("--hard-from", type=int, default=None)
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--cache", default=str(CACHE_PATH))
    parser.add_argument("--save-every", type=int, default=0, help="default: every 10%% of --steps")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.arch == "v1":
        if args.normalize or args.readout != "last" or args.no_batchnorm or args.distance == "sqeuclidean":
            parser.error("--arch v1 is v1's fixed network: no TypeNet flags, cosine distance only")
        args.distance = "cosine"
    args.distance = args.distance or "sqeuclidean"
    if args.distance == "cosine" and args.arch == "typenet" and not args.normalize:
        parser.error("cosine distance is 1 - a.b, which needs unit vectors: add --normalize")
    args.margin = args.margin if args.margin is not None else (MARGIN if args.distance == "sqeuclidean" else COSINE_MARGIN)
    args.save_every = args.save_every or args.steps // 10

    run_dir = RUNS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampler = make_sampler(args.split, args.cache)
    model, model_kwargs = build_model(args)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    start_step = 0
    if args.resume:
        start_step = load_for_resume(args.resume, model, optimizer, device)
        # Reseeding with --seed alone would replay the triplet draws of steps 1..start_step (seen in R1).
        np.random.seed(args.seed + start_step)
        print(f"Resuming from {args.resume} at step {start_step}", flush=True)

    config = {
        **vars(args),
        "model": model_kwargs,
        "num_train_subjects": len(sampler.allowed_subjects),
    }
    print(f"config: {config}", flush=True)

    def checkpoint(step: int, path: Path):
        save_checkpoint(model, {**config, "steps": step, "triplets_seen": step * args.batch_size},
                        path, optimizer=optimizer, step=step)
        print(f"saved {path}", flush=True)

    sums, tic = np.zeros(3), time.time()
    hard_start_spread, collapse_run = None, 0
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        batch = to_tensors(sampler.sample_batch(args.batch_size), device)
        mode = negatives_mode(step, args.semi_from, args.hard_from)
        loss, emb_norm, spread = train_step(model, optimizer, batch, args.arch, mode, args.margin, args.distance)
        if not np.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")

        if mode == "hard":
            if hard_start_spread is None:
                # Hardest-mined loss / margin on the first hardest batch. Collapsing to one point makes the loss
                # exactly the margin, so a ratio > 1 means collapse lowers the loss (pilot finding).
                hard_start_spread = spread
                print(f"HARD-START step {step} ratio={loss / args.margin:.3f} spread={spread:.4f}", flush=True)
            collapsed = abs(loss - args.margin) <= 0.002 * args.margin and spread < 0.01 * hard_start_spread
            collapse_run = collapse_run + 1 if collapsed else 0
            if collapse_run >= COLLAPSE_STEPS:
                print(f"COLLAPSED at step {step}: loss {loss:.4f} = margin, spread {spread:.5f} "
                      f"(was {hard_start_spread:.4f})", flush=True)
                checkpoint(step, run_dir / "final.pt")
                return

        sums += (loss, emb_norm, spread)
        if step % LOG_EVERY == 0:
            mean_loss, mean_norm, mean_spread = sums / LOG_EVERY
            print(f"Step {step}/{args.steps}, loss {mean_loss:.4f}, {mode}, norm {mean_norm:.3f}, "
                  f"spread {mean_spread:.4f}, {(time.time() - tic) / LOG_EVERY:.3f} s/step", flush=True)
            sums, tic = np.zeros(3), time.time()

        if step % args.save_every == 0 and step != args.steps:
            checkpoint(step, run_dir / f"step{step}.pt")

    checkpoint(args.steps, run_dir / "final.pt")


if __name__ == "__main__":
    main()
