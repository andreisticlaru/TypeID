"""TypeNet's own evaluation protocol (Acien et al. 2021, Sec 5.1 and 5.2), for like-for-like comparison.

Per person: 15 sequences in time order (ascending session id). Gallery = the first G (<=10),
query = the last 5. A sequence is ONE 50-keystroke window: the first window of a session.
The score between a gallery person and a query is the mean Euclidean distance over all
gallery x query embedding pairs.

  Identification (their Table 5): a query person is ranked against a background of B=1,000
      gallery people (10 gallery, 5 query sequences each); Rank-N by smallest score.
  Authentication (their Table 2): 5 genuine scores per person, plus one impostor query from
      every other person; EER is computed per person, then averaged.

Differs from eval/rank_n.py, which pools every window of a session and samples sessions at random.
Only people with 15 or more usable sessions are eligible (~3.8k of the held-out subjects, because
sessions under the 25-keystroke floor are dropped from the cache).

run with: python -m eval.typenet_protocol --checkpoint model/encoder_hard.pt --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from eval.rank_n import embed_rows, load_encoder, load_eval_data
from model.train import CHECKPOINT_PATH

SEQUENCES, GALLERY_MAX, QUERIES = 15, 10, 5
ENROLLMENT_SIZES = (1, 2, 5, 7, 10)  # G values in TypeNet's Table 2


def session_embeddings(model, data, device) -> dict[str, torch.Tensor]:
    """Per eligible subject, an (n_sessions, D) tensor: the first-window embedding of each session, in time order."""
    windows, mask, eval_rows, rows_by_session = data
    embeddings = torch.from_numpy(embed_rows(model, windows, mask, eval_rows, device)).to(device)
    sessions_by_subject: dict[str, list] = {}
    for subject, session in rows_by_session:
        sessions_by_subject.setdefault(subject, []).append(session)
    return {
        subject: embeddings[[np.searchsorted(eval_rows, rows_by_session[(subject, s)][0]) for s in sorted(sessions, key=int)]]
        for subject, sessions in sessions_by_subject.items()
        if len(sessions) >= SEQUENCES
    }


def draw_background(embeddings, seed, size) -> torch.Tensor:
    """(size, 15, D): a random background set, each person's first 15 sequences."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(sorted(embeddings), size=min(size, len(embeddings)), replace=False)
    return torch.stack([embeddings[s][:SEQUENCES] for s in chosen])


def mean_pairwise_distance(gallery: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """(N, g, D) x (N, q, D) -> (N, N): entry [i, j] = gallery person i vs query person j."""
    n, g, d = gallery.shape
    q = query.shape[1]
    return torch.cdist(gallery.reshape(n * g, d), query.reshape(n * q, d)).reshape(n, g, n, q).mean(dim=(1, 3))


def identification_ranks(background: torch.Tensor) -> torch.Tensor:
    distances = mean_pairwise_distance(background[:, :GALLERY_MAX], background[:, GALLERY_MAX:])
    return 1 + (distances < distances.diag()[None, :]).sum(dim=0)


def subject_eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    thresholds = np.sort(np.concatenate([genuine, impostor]))
    frr = (genuine[None, :] > thresholds[:, None]).mean(axis=1)
    far = (impostor[None, :] <= thresholds[:, None]).mean(axis=1)
    k = np.argmin(np.abs(far - frr))
    return float((far[k] + frr[k]) / 2)


def authentication_eer(background: torch.Tensor, gallery_size: int, rng) -> float:
    """Mean per-person EER (%) with `gallery_size` enrollment sequences and k = len(background) people."""
    n = background.shape[0]
    gallery, query = background[:, :gallery_size], background[:, GALLERY_MAX:]
    genuine = torch.stack([torch.cdist(gallery[i], query[i]).mean(dim=0) for i in range(n)]).cpu().numpy()
    one_query = query[torch.arange(n), torch.from_numpy(rng.integers(0, QUERIES, size=n))]  # one impostor sample per person
    impostor = torch.cdist(gallery.reshape(n * gallery_size, -1), one_query).reshape(n, gallery_size, n).mean(dim=1).cpu().numpy()
    return 100 * float(np.mean([subject_eer(genuine[i], np.delete(impostor[i], i)) for i in range(n)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", default=[str(CHECKPOINT_PATH)])
    parser.add_argument("--background", type=int, default=1000, help="people per test (TypeNet: 1,000)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--untrained", action="store_true", help="random-weights baseline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_eval_data()
    runs = [("untrained", None)] if args.untrained else [(Path(c).name, c) for c in args.checkpoint]

    print(f"TypeNet protocol, background={args.background}, {len(args.seeds)} seed(s); mean +- std across seeds")
    print(f"{'checkpoint':<28}{'Rank-1':>14}{'Rank-5':>14}{'Rank-20':>14}   EER% at G=" + "/".join(map(str, ENROLLMENT_SIZES)))
    for label, path in runs:
        embeddings = session_embeddings(load_encoder(path, args.untrained, device), data, device)
        ranks, eers = [], {g: [] for g in ENROLLMENT_SIZES}
        for seed in args.seeds:
            background = draw_background(embeddings, seed, args.background)
            ranks.append(identification_ranks(background).cpu().numpy())
            rng = np.random.default_rng(1000 + seed)
            for g in ENROLLMENT_SIZES:
                eers[g].append(authentication_eer(background, g, rng))
        cells = ""
        for n in (1, 5, 20):
            accuracy = np.array([(r <= n).mean() for r in ranks]) * 100
            cells += f"{accuracy.mean():>8.1f} +-{accuracy.std(ddof=1) if len(accuracy) > 1 else 0:<3.1f}"
        print(f"{label:<28}{cells}   " + "/".join(f"{np.mean(v):.2f}" for v in eers.values()))
    print(f"eligible people (>=15 sessions): {len(embeddings)}, chance Rank-1 = {1 / min(args.background, len(embeddings)):.2%}")


if __name__ == "__main__":
    main()
