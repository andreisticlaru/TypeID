"""Open-set identification eval: enroll held-out subjects, then identify them.

Simulates the live system offline on people the model never saw in training:
  1. Pick `num_subjects` held-out subjects (from data/split.json) with enough sessions.
  2. Per subject: `enroll_sessions` sessions form the gallery entry (embeddings
     averaged, like enrolling with 2-3 prompts); one further, different session
     is the probe (a sentence never used for enrollment, so we test rhythm, not content).
  3. Rank every gallery entry by cosine similarity to each probe.
  4. Rank-N accuracy = fraction of probes whose true identity is in the top N.

run with: python -m eval.rank_n --num-subjects 1000
Add --untrained for a random-weights baseline (should score near chance).
"""

from __future__ import annotations

import argparse
import json

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


def select_sessions(subject_ids, session_ids, eval_subjects, num_subjects, enroll_sessions, rng):
    """Return {subject: (enroll_session_ids, probe_session_id)} for sampled eval subjects."""
    candidate_rows = np.flatnonzero(np.isin(subject_ids, list(eval_subjects)))
    sessions_by_subject: dict[str, set] = {}
    for i in candidate_rows:
        sessions_by_subject.setdefault(subject_ids[i], set()).add(session_ids[i])

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


def evaluate(checkpoint_path=CHECKPOINT_PATH, num_subjects=1000, enroll_sessions=3, seed=0, untrained=False):
    """Return an int array: for each probe, the 1-based rank of its true identity."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_encoder(checkpoint_path, untrained, device)

    windows = np.load(PREPROCESSED_PATH / "windows.npy")
    mask = np.load(PREPROCESSED_PATH / "mask.npy")
    subject_ids = np.load(PREPROCESSED_PATH / "subject_ids.npy")
    session_ids = np.load(PREPROCESSED_PATH / "session_ids.npy")
    eval_subjects = set(json.load(open(DATA_ROOT / "split.json"))["eval_subjects"])

    rng = np.random.default_rng(seed)
    plan = select_sessions(subject_ids, session_ids, eval_subjects, num_subjects, enroll_sessions, rng)

    wanted = {(s, sess) for s, (enroll, probe) in plan.items() for sess in [*enroll, probe]}
    subject_rows = np.flatnonzero(np.isin(subject_ids, list(plan)))
    rows_by_session: dict[tuple, list[int]] = {}
    for i in subject_rows:
        key = (subject_ids[i], session_ids[i])
        if key in wanted:
            rows_by_session.setdefault(key, []).append(i)

    all_rows = np.array(sorted(r for rows in rows_by_session.values() for r in rows))
    embeddings = embed_rows(model, windows, mask, all_rows, device)
    emb_of_row = {r: e for r, e in zip(all_rows, embeddings)}

    subjects = list(plan)
    gallery, probes = [], []
    for s in subjects:
        enroll, probe = plan[s]
        enroll_embs = [emb_of_row[r] for sess in enroll for r in rows_by_session[(s, sess)]]
        gallery.append(pooled_embedding(np.stack(enroll_embs)))
        probes.append(pooled_embedding(np.stack([emb_of_row[r] for r in rows_by_session[(s, probe)]])))

    scores = np.stack(probes) @ np.stack(gallery).T  # (num_probes, num_gallery), cosine similarity
    true_scores = np.diag(scores)[:, None]
    return 1 + (scores > true_scores).sum(axis=1)


def rank_n_accuracy(ranks: np.ndarray, n: int) -> float:
    return float((ranks <= n).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--num-subjects", type=int, default=1000)
    parser.add_argument("--enroll-sessions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--untrained", action="store_true", help="random-weights baseline")
    args = parser.parse_args()

    ranks = evaluate(args.checkpoint, args.num_subjects, args.enroll_sessions, args.seed, args.untrained)

    label = "untrained baseline" if args.untrained else args.checkpoint
    print(f"{label}: {len(ranks)} gallery subjects, {args.enroll_sessions} enroll sessions each")
    for n in RANKS_REPORTED:
        print(f"  Rank-{n:<3d} {rank_n_accuracy(ranks, n):6.1%}")
    print(f"  chance Rank-1 = {1 / len(ranks):.2%}, median rank = {int(np.median(ranks))}")


if __name__ == "__main__":
    main()
