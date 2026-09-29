"""Train TypeNet on the train-subject split, with the paper's random triplets or in-batch mining.

  paper recipe:  python -u -m typenet.train --steps 30000 --save-every 10000 --out typenet/checkpoints/typenet_a.pt
  lr 1e-3:       ... --lr 1e-3 --out typenet/checkpoints/typenet_lr1e-3.pt
  + mining:      ... --lr 1e-3 --mine semi --mine-from 10000
                     --resume typenet/checkpoints/typenet_lr1e-3_step10000.pt --out typenet/checkpoints/typenet_semi.pt

Paper recipe (Sec 4.3): Adam lr 0.05, betas (0.9, 0.999), eps 1e-8; 512 random triplets per batch;
200 epochs x 150 batches = 30,000 steps. The lr is constant, so the first 30k steps of a longer run ARE
the paper's run, and its step-30000 checkpoint is the faithful reproduction.

Same data path as model/train.py (TripletSampler on data/split.json's train subjects), so the two
models differ only in network, loss and recipe.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from model.train import build_sampler, load_for_resume, save_checkpoint, to_tensors
from typenet.losses import MARGIN, mined_loss, triplet_loss
from typenet.network import TypeNetEncoder, INPUT_SIZE, HIDDEN_SIZE, DROPOUT, RECURRENT_DROPOUT, INPUT_SCALE

BATCH_SIZE = 512
LR = 0.05
LOG_EVERY = 100


def train_step(model, optimizer, batch, mine: str | None) -> tuple[float, float]:
    """Returns (loss, mean embedding norm). The norm is logged because a collapse onto one point (loss
    stuck at the margin, norm and spread shrinking) is the known failure of hardest-negative mining."""
    # One pass over anchor+positive+negative; groups=3 keeps BatchNorm statistics per branch, as a Keras
    # siamese model computes them (tested equal to three separate calls).
    x = torch.cat([batch["anchor_windows"], batch["positive_windows"], batch["negative_windows"]])
    mask = torch.cat([batch["anchor_masks"], batch["positive_masks"], batch["negative_masks"]])
    anchor, positive, negative = model(x, mask, groups=3).chunk(3)

    if mine:
        pool_subjects = np.concatenate([batch["anchor_subjects"], batch["negative_subjects"]])
        same_subject = torch.from_numpy(batch["anchor_subjects"][:, None] == pool_subjects[None, :]).to(anchor.device)
        loss = mined_loss(anchor, positive, negative, same_subject, mine)
    else:
        loss = triplet_loss(anchor, positive, negative)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), anchor.detach().norm(dim=1).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000, help="total steps (with --resume, the target, not extra)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--mine", choices=["semi", "hard"], default=None,
                        help="in-batch negative mining (default: none, random triplets as in the paper)")
    parser.add_argument("--mine-from", type=int, default=0, help="mine only for steps after this one")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="typenet/checkpoints/typenet_a.pt")
    args = parser.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampler, split_seed = build_sampler()
    model = TypeNetEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    start_step = 0
    if args.resume:
        start_step = load_for_resume(args.resume, model, optimizer, device)
        print(f"Resuming from {args.resume} at step {start_step}")

    config = {
        "arch": "typenet",
        "model": {"input_size": INPUT_SIZE, "hidden_size": HIDDEN_SIZE, "dropout": DROPOUT,
                  "recurrent_dropout": RECURRENT_DROPOUT, "input_scale": INPUT_SCALE},
        "margin": MARGIN,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "mine": args.mine,
        "mine_from": args.mine_from if args.mine else None,
        "seed": args.seed,
        "split_seed": split_seed,
        "num_train_subjects": len(sampler.allowed_subjects),
        "resumed_from": args.resume,
    }

    running, norm, tic = 0.0, 0.0, time.time()
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        batch = to_tensors(sampler.sample_batch(args.batch_size), device)
        mine = args.mine if step > args.mine_from else None
        loss, emb_norm = train_step(model, optimizer, batch, mine)
        if not np.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")

        running += loss
        norm += emb_norm
        if step % LOG_EVERY == 0:
            print(f"Step {step}/{args.steps}, loss {running / LOG_EVERY:.4f}, {mine or 'random'}, "
                  f"norm {norm / LOG_EVERY:.2f}, {(time.time() - tic) / LOG_EVERY:.3f} s/step", flush=True)
            running, norm, tic = 0.0, 0.0, time.time()

        if args.save_every and step % args.save_every == 0 and step != args.steps:
            save_checkpoint(model, {**config, "steps": step},
                            out_path.with_name(f"{out_path.stem}_step{step}{out_path.suffix}"),
                            optimizer=optimizer, step=step)

    save_checkpoint(model, config, out_path, optimizer=optimizer, step=args.steps)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
