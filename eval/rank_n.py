"""Open-set identification eval: enroll held-out subjects, then identify them.

Simulates the live system offline on people the model never saw in training:
  1. Pick `num_subjects` held-out subjects (from data/split.json) with enough sessions.
  2. Per subject: `enroll_sessions` sessions form the gallery entry (embeddings
     averaged, like enrolling with 2-3 prompts); one further, different session
     is the probe (a sentence never used for enrollment, so we test rhythm, not content).
  3. Rank every gallery entry by cosine similarity to each probe.
  4. Rank-N accuracy = fraction of probes whose true identity is in the top N.

One seed = one random draw of subjects and sessions, which is noisy (~1-2 points).
Pass several --seeds and read the mean +- std. The same seed picks the same
subjects/sessions for every checkpoint, so checkpoints are compared like for like.

run with: python -m eval.rank_n --seeds 0 1 2 3 4
Compare checkpoints: python -m eval.rank_n --checkpoint model/encoder_step90000.pt model/encoder.pt --seeds 0 1 2 3 4
Add --untrained for a random-weights baseline (should score near chance).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.config import DATA_ROOT, PREPROCESSED_PATH
from model.network import KeystrokeEncoder
from model.train import CHECKPOINT_PATH

RANKS_REPORTED = (1, 5, 10, 20)


def load_encoder(checkpoint_path, untrained: bool, device: torch.device) -> KeystrokeEncoder:
    if untrained:
        torch.manual_seed(0)
        model = KeystrokeEncoder()
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model = KeystrokeEncoder(**ckpt["config"]["model"])
        model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()  # eval(): dropout off


def load_eval_data():
    """Load the cache once; index only the held-out (eval) subjects' rows by (subject, session)."""
    windows = np.load(PREPROCESSED_PATH / "windows.npy")
    mask = np.load(PREPROCESSED_PATH / "mask.npy")
    subject_ids = np.load(PREPROCESSED_PATH / "subject_ids.npy")
    session_ids = np.load(PREPROCESSED_PATH / "session_ids.npy")
    eval_subjects = set(json.load(open(DATA_ROOT / "split.json"))["eval_subjects"])

    eval_rows = np.flatnonzero(np.isin(subject_ids, list(eval_subjects)))  # ascending
    rows_by_session: dict[tuple, list[int]] = {}
    for i in eval_rows:
        rows_by_session.setdefault((subject_ids[i], session_ids[i]), []).append(i)
    return windows, mask, eval_rows, rows_by_session


def select_sessions(rows_by_session, num_subjects, enroll_sessions, rng):
    """Return {subject: (enroll_session_ids, probe_session_id)} for sampled eval subjects."""
    sessions_by_subject: dict[str, list] = {}
    for subject, session in rows_by_session:
        sessions_by_subject.setdefault(subject, []).append(session)

    eligible = sorted(s for s, sess in sessions_by_subject.items() if len(sess) >= enroll_sessions + 1)
    chosen = rng.choice(eligible, size=min(num_subjects, len(eligible)), replace=False)

    plan = {}
    for subject in chosen:
        sessions = sorted(sessions_by_subject[subject])
        rng.shuffle(sessions)
        plan[subject] = (sessions[:enroll_sessions], sessions[enroll_sessions])
    return plan


@torch.no_grad()
def embed_rows(model, windows, mask, rows, device, batch_size=1024) -> np.ndarray:
    out = []
    for start in range(0, len(rows), batch_size):
        idx = rows[start:start + batch_size]
        x = torch.from_numpy(windows[idx]).to(device)
        m = torch.from_numpy(mask[idx]).to(device)
        out.append(model(x, m).cpu().numpy())
    return np.concatenate(out)


def pooled_embedding(embeddings: np.ndarray) -> np.ndarray:
    """Mean of window embeddings, re-normalized so dot product stays cosine similarity."""
    mean = embeddings.mean(axis=0)
    return mean / np.linalg.norm(mean)


def ranks_for_seed(embeddings, eval_rows, rows_by_session, seed, num_subjects, enroll_sessions) -> np.ndarray:
    """For each probe, the 1-based rank of its true identity among the gallery."""
    plan = select_sessions(rows_by_session, num_subjects, enroll_sessions, np.random.default_rng(seed))

    def pooled(subject, sessions):
        rows = np.concatenate([rows_by_session[(subject, s)] for s in sessions])
        return pooled_embedding(embeddings[np.searchsorted(eval_rows, rows)])

    gallery = np.stack([pooled(s, enroll) for s, (enroll, _) in plan.items()])
    probes = np.stack([pooled(s, [probe]) for s, (_, probe) in plan.items()])

    scores = probes @ gallery.T  # (num_probes, num_gallery), cosine similarity
    true_scores = np.diag(scores)[:, None]
    return 1 + (scores > true_scores).sum(axis=1)


def evaluate_model(model, data, device, seeds, num_subjects=1000, enroll_sessions=3) -> list[np.ndarray]:
    """One ranks array per seed. Embeds all held-out windows once, then reuses them for every seed."""
    windows, mask, eval_rows, rows_by_session = data
    embeddings = embed_rows(model, windows, mask, eval_rows, device)
    return [
        ranks_for_seed(embeddings, eval_rows, rows_by_session, seed, num_subjects, enroll_sessions)
        for seed in seeds
    ]


def rank_n_accuracy(ranks: np.ndarray, n: int) -> float:
    return float((ranks <= n).mean())


def summarize(ranks_per_seed: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std (across seeds) of Rank-N accuracy for each N in RANKS_REPORTED."""
    table = np.array([[rank_n_accuracy(r, n) for n in RANKS_REPORTED] for r in ranks_per_seed])
    std = table.std(axis=0, ddof=1) if len(table) > 1 else np.zeros(table.shape[1])
    return table.mean(axis=0), std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", default=[str(CHECKPOINT_PATH)])
    parser.add_argument("--num-subjects", type=int, default=1000)
    parser.add_argument("--enroll-sessions", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--untrained", action="store_true", help="random-weights baseline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_eval_data()

    runs = [("untrained", None)] if args.untrained else [(Path(c).name, c) for c in args.checkpoint]

    print(f"gallery={args.num_subjects} subjects, {args.enroll_sessions} enroll sessions, "
          f"{len(args.seeds)} seed(s); Rank-N accuracy % (mean  +- std across seeds)")
    print(f"{'checkpoint':<26}" + "".join(f"{'Rank-' + str(n):>16}" for n in RANKS_REPORTED))
    for label, path in runs:
        model = load_encoder(path, args.untrained, device)
        ranks_per_seed = evaluate_model(model, data, device, args.seeds, args.num_subjects, args.enroll_sessions)
        mean, std = summarize(ranks_per_seed)
        print(f"{label:<26}" + "".join(f"{100 * m:>9.1f}  +- {100 * s:<4.1f}" for m, s in zip(mean, std)))
    print(f"chance Rank-1 = {1 / args.num_subjects:.2%}")


if __name__ == "__main__":
    main()
