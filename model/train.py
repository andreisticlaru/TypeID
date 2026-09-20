"""Train the KeystrokeEncoder with triplet loss on the train-subject split.

run with: .venv-model/Scripts/python -m model.train --steps 20000

Data flow per step:
    TripletSampler.sample_batch  -> numpy (anchor/positive/negative windows + masks)
    to_tensors                   -> torch tensors on the device
    KeystrokeEncoder (x3 passes, SAME weights) -> three (batch, 128) embeddings
    triplet_loss                 -> scalar
    backward + Adam step         -> weights update
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.config import DATA_ROOT, PREPROCESSED_PATH
from model.losses import MARGIN, triplet_loss
from model.network import KeystrokeEncoder, INPUT_SIZE, HIDDEN_SIZE, EMBEDDING_DIM, DROPOUT
from model.triplet_sampler import TripletSampler

BATCH_SIZE = 64
LR = 1e-3
LOG_EVERY = 100
CHECKPOINT_PATH = DATA_ROOT.parent / "model" / "encoder.pt"


def build_sampler() -> tuple[TripletSampler, int]:
    """TripletSampler restricted to the train subjects from data/split.json.

    Returns (sampler, split_seed) so the seed can be recorded in the checkpoint.
    Never pass eval subjects here: the held-out set must stay unseen.
    """
    # TODO: load DATA_ROOT / "split.json", take split["train_subjects"] as a set,
    # and construct TripletSampler with the four .npy paths under
    # PREPROCESSED_PATH plus subjects=<that set>.
    
    DATA_SPLIT_PATH = DATA_ROOT / "split.json"
    with open(DATA_SPLIT_PATH, "r") as f:
        split = json.load(f)
    train_subjects = set(split["train_subjects"])

    sampler = TripletSampler(
        windows_path=PREPROCESSED_PATH / "windows.npy",
        mask_path=PREPROCESSED_PATH / "mask.npy",
        subject_ids_path=PREPROCESSED_PATH / "subject_ids.npy",
        session_ids_path=PREPROCESSED_PATH / "session_ids.npy",
        subjects=train_subjects)

    return sampler, split["seed"]



def to_tensors(batch: dict, device: torch.device) -> dict:
    """Convert the sampler's numpy batch into torch tensors on `device`.

    Windows -> float32, masks -> bool (KeystrokeEncoder.forward expects both).
    Subject-id arrays are strings, so they stay as numpy; mining compares them
    on the CPU.
    """
    for key, value in batch.items():
        if value.dtype.kind in "fb":
            batch[key] = torch.from_numpy(value).to(device)

    return batch


def mine_semi_hard(anchor, positive, negative, same_subject, margin: float = MARGIN):
    """Swap each anchor's sampled negative for a harder one already in the batch.

    Candidates are every positive and negative embedding in the batch (2B of
    them), minus any belonging to the anchor's own subject -- without that mask
    a same-subject collision would be the *closest* candidate and get picked
    preferentially, training the model to push one person's samples apart.

    Preference is semi-hard: the closest candidate still farther than the
    positive. That yields a non-zero loss without the collapse risk of always
    taking the outright hardest negative. If nothing qualifies (every candidate
    is already closer than the positive), fall back to the farthest valid one,
    the gentlest of the violating options.

    Returns (d_ap, d_an) so the caller can form the loss.
    """
    pool = torch.cat([positive, negative])                # (2B, D)
    d_ap = 1 - (anchor * positive).sum(1, keepdim=True)   # (B, 1)
    d_an = 1 - anchor @ pool.T                            # (B, 2B)

    valid = d_an.detach().masked_fill(same_subject, float("inf"))
    semi = valid.masked_fill(valid <= d_ap.detach(), float("inf"))
    idx = semi.argmin(1)

    none_qualify = semi.gather(1, idx[:, None]).isinf().squeeze(1)
    if none_qualify.any():
        farthest = valid.masked_fill(valid.isinf(), -float("inf")).argmax(1)
        idx = torch.where(none_qualify, farthest, idx)

    return d_ap.squeeze(1), d_an.gather(1, idx[:, None]).squeeze(1)

def train_step(model: KeystrokeEncoder, optimizer: torch.optim.Optimizer, batch: dict, mine: bool = False) -> float:
    """One optimization step on a batch of tensors. Returns the loss as a float.

    Note: anchor, positive and negative all go through the SAME model instance --
    this is one set of weights, run three times.

    `model(x, mask)` works because nn.Module defines __call__, which does
    train/eval and hook bookkeeping and then calls KeystrokeEncoder.forward.
    Batch keys from TripletSampler.sample_batch are plural ("anchor_windows",
    "anchor_masks", ...); the singular names belong to sample_triplet.
    """
    # TODO:
    #   1. anchor   = model(batch["anchor_windows"],   batch["anchor_masks"])
    #      positive = ...                                (same for negative)
    #   2. loss = triplet_loss(anchor, positive, negative)
    #   3. optimizer.zero_grad(); loss.backward(); optimizer.step()
    #   4. return loss.item()
    
    # Step 1: Forward Pass anchor, positive, and negative through the model
    anchor = model(batch["anchor_windows"], batch["anchor_masks"])
    positive = model(batch["positive_windows"], batch["positive_masks"])
    negative = model(batch["negative_windows"], batch["negative_masks"])

    # Step 2: Compute the triplet loss
    if mine:
        pool_subjects = np.concatenate([batch["anchor_subjects"], batch["negative_subjects"]])
        same_subject = torch.from_numpy(
            batch["anchor_subjects"][:, None] == pool_subjects[None, :]
        ).to(anchor.device)
        d_ap, d_an = mine_semi_hard(anchor, positive, negative, same_subject)
        loss = torch.clamp(d_ap - d_an + MARGIN, min=0).mean()
    else:
        loss = triplet_loss(anchor, positive, negative)

    # Step 3: Backward Pass and Optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Step 4: Return the loss as a float
    return loss.item()


    


def save_checkpoint(
    model: KeystrokeEncoder,
    config: dict,
    path=CHECKPOINT_PATH,
    optimizer: torch.optim.Optimizer | None = None,
    step: int | None = None,
) -> None:
    """Save the encoder weights (state_dict, not the pickled module) plus `config`.

    Bundled in one file so the weights can never be separated from the settings
    that produced them. Load with:
        ckpt = torch.load(path, map_location="cpu")
        model = KeystrokeEncoder(**ckpt["config"]["model"])
        model.load_state_dict(ckpt["state_dict"])

    `optimizer` and `step` are only needed to resume training: Adam's running
    averages live in the optimizer, and without them a resumed run restarts
    with empty momentum history.
    """
    ckpt = {"state_dict": model.state_dict(), "config": config}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if step is not None:
        ckpt["step"] = step
    torch.save(ckpt, path)


def load_for_resume(path, model: KeystrokeEncoder, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    """Load weights (and Adam state, if saved) into `model`/`optimizer`; return the step reached.

    Checkpoints from before resume support have no optimizer state or "step"
    key: they resume with fresh Adam history, and the step falls back to the
    config's total steps.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", ckpt["config"]["steps"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--save-every", type=int, default=0,
                        help="also save encoder_step<N>.pt every N steps (0 = only save at the end)")
    parser.add_argument("--resume", default=None,
                        help="checkpoint to continue from; --steps is then the TOTAL target, not extra steps")
    parser.add_argument("--mine", action="store_true",
                        help="semi-hard in-batch negative mining instead of the randomly sampled negative")
    parser.add_argument("--out", default=str(CHECKPOINT_PATH),
                        help="where to write the final checkpoint; --save-every files get a _step<N> suffix")
    args = parser.parse_args()
    out_path = Path(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: build the sampler, then model = KeystrokeEncoder().to(device) and
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr).
    #
    # Then loop for args.steps:
    #   - model.train()   (enables dropout)
    #   - batch = to_tensors(sampler.sample_batch(args.batch_size), device)
    #   - loss = train_step(model, optimizer, batch)
    #   - every LOG_EVERY steps, print step and a running mean of the loss
    # Finally save_checkpoint(model, config) with a config dict like:
    #   {"model": {"input_size": INPUT_SIZE, "hidden_size": HIDDEN_SIZE,
    #              "embedding_dim": EMBEDDING_DIM, "dropout": DROPOUT},   # network.py constants
    #    "margin": MARGIN,                                                # losses.py
    #    "lr": args.lr, "batch_size": args.batch_size, "steps": args.steps,
    #    "split_seed": split["seed"], "num_train_subjects": <len(train_subjects)>}
    # (the "model" sub-dict must match KeystrokeEncoder's __init__ kwargs.)

    sampler, split_seed = build_sampler()
    model = KeystrokeEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume:
        start_step = load_for_resume(args.resume, model, optimizer, device)
        for group in optimizer.param_groups:
            group["lr"] = args.lr  # the loaded optimizer state carries the old lr; --lr wins
        if start_step >= args.steps:
            print(f"Checkpoint is already at step {start_step} >= --steps {args.steps}; nothing to do.")
            return
        print(f"Resuming from {args.resume} at step {start_step}")

    config = {
        "model": {
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "dropout": DROPOUT,
        },
        "margin": MARGIN,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "split_seed": split_seed,
        "num_train_subjects": len(sampler.allowed_subjects),
        "mine": args.mine,
        "resumed_from": args.resume,
    }

    running_loss = 0.0
    for step in range(start_step + 1, args.steps + 1):
        model.train()  # Enable dropout
        batch = to_tensors(sampler.sample_batch(args.batch_size), device)
        loss = train_step(model, optimizer, batch, mine=args.mine)

        running_loss += loss
        if step % LOG_EVERY == 0:
            avg_loss = running_loss / LOG_EVERY
            print(f"Step {step}/{args.steps}, Average Loss: {avg_loss:.4f}")
            running_loss = 0.0

        if args.save_every and step % args.save_every == 0 and step != args.steps:
            save_checkpoint(
                model, {**config, "steps": step},
                out_path.with_name(f"{out_path.stem}_step{step}{out_path.suffix}"),
                optimizer=optimizer, step=step,
            )

    save_checkpoint(model, config, out_path, optimizer=optimizer, step=args.steps)
 
    

if __name__ == "__main__":
    main()
