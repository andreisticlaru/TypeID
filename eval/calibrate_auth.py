"""Calibrate GLOBAL decision thresholds for authentication and identification, per prompt count.

Why this exists separately from eval/typenet_protocol.py: that script reports TypeNet's
per-subject EER, where the operating point is chosen from each person's own genuine and
impostor scores. That measures separability and is the right thing for a like-for-like
comparison with the paper, but it is an oracle -- a deployed system has to commit to ONE
threshold in advance, before it has ever seen the claimant's score distribution. The
per-subject number is therefore optimistic as a deployment figure, and can't be lifted into
the backend.

The UI lets people enrol from E and query with Q prompts, E, Q in {1, 5, 10}. Both change the
score distributions (a 10-prompt average is steadier than one prompt), so each (E, Q) pair gets
its own threshold. Everything is scored exactly the way the backend does it:

  session  = pooled embedding of one prompt's windows                     (embedding.py)
  template = mean of E session embeddings                                 (embedding.embed_sessions)
  query    = mean of Q session embeddings                                 (embedding.embed_sessions)
  score    = cosine(query, template)                                      (identify.py)

Authentication (1:1): genuine = query vs own template, impostor = query vs every other subject's
template. Identification (open set): a gallery of templates, probes from people in it and people
not in it; the decision is whether the TOP score clears the threshold. The best of hundreds of
impostors scores far higher than a single one, which is why identification needs its own
threshold rather than the authentication one. Its threshold is set by a cap on strangers being
named (FPIR), not at the EER: naming the wrong person is the costly mistake for "who typed this?".

Subjects are held-out (split.json eval_subjects), so no identity here was seen in training.
Aalto gives each person at most 15 usable sessions, so a cell is only measurable when
E + Q <= 15: (10, 10) is not, and the backend falls back to (10, 5) for it.

run with: python -m eval.calibrate_auth --checkpoint model/encoder_hard.pt
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from eval.rank_n import embed_rows, load_encoder, load_eval_data, pooled_embedding
from model.train import CHECKPOINT_PATH

PROMPT_COUNTS = (1, 5, 10)
IDENTIFY_GALLERY = 500  # matches the live gallery's Aalto background (seed_aalto_gallery.py)
IDENTIFY_ENROLL = 5  # the background is enrolled from 5 prompts, so it sets the top-score scale


def probes_per_subject(query_prompts: int) -> int:
    # Several single-prompt probes per person are cheap; a 5- or 10-prompt probe eats a third of
    # someone's sessions or more, so those cells get one probe each.
    return 3 if query_prompts == 1 else 1


def session_embedding_table(model, data, device):
    """{subject: (n_sessions, D)} -- one pooled embedding per session, backend-identical pooling."""
    windows, mask, eval_rows, rows_by_session = data
    all_embeddings = embed_rows(model, windows, mask, eval_rows, device)
    row_index = {row: i for i, row in enumerate(eval_rows)}

    by_subject: dict[str, list] = {}
    for (subject, session), rows in rows_by_session.items():
        by_subject.setdefault(subject, []).append(
            pooled_embedding(all_embeddings[[row_index[r] for r in rows]])
        )
    return {s: np.stack(v) for s, v in by_subject.items()}


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def split_sessions(sessions, enroll, query, probes, rng):
    """Disjoint sessions: one E-session template, `probes` Q-session queries (both unit norm)."""
    order = rng.permutation(len(sessions))
    template = _unit(sessions[order[:enroll]].mean(axis=0))
    rest = order[enroll:]
    queries = _unit(np.stack([sessions[rest[k * query:(k + 1) * query]].mean(axis=0) for k in range(probes)]))
    return template, queries


def eligible(table, need, count, rng):
    subjects = sorted(s for s, v in table.items() if len(v) >= need)
    return list(rng.choice(subjects, size=min(count, len(subjects)), replace=False))


def authentication_scores(table, enroll, query, subjects, seed):
    """(genuine, impostor) cosine scores for one (E, Q) cell."""
    rng = np.random.default_rng(seed)
    probes = probes_per_subject(query)
    chosen = eligible(table, enroll + probes * query, subjects, rng)
    templates, queries = zip(*(split_sessions(table[s], enroll, query, probes, rng) for s in chosen))
    t, q = np.stack(templates), np.stack(queries)  # (n, D), (n, P, D)
    scores = np.einsum("npd,md->npm", q, t)  # [i, k, j] = subject i's probe k vs subject j's template

    n = len(chosen)
    own = np.arange(n)
    genuine = scores[own, :, own].ravel()
    impostor = scores[~np.eye(n, dtype=bool)[:, None, :].repeat(probes, axis=1)].ravel()
    return genuine, impostor, n


def identification_scores(table, query, outsiders, seed):
    """Open-set top scores: (enrolled probes' top score, outsiders' top score) against the gallery."""
    rng = np.random.default_rng(seed)
    chosen = eligible(table, IDENTIFY_ENROLL + query, IDENTIFY_GALLERY + outsiders, rng)
    members, strangers = chosen[:IDENTIFY_GALLERY], chosen[IDENTIFY_GALLERY:]

    templates, member_queries = zip(*(split_sessions(table[s], IDENTIFY_ENROLL, query, 1, rng) for s in members))
    gallery = np.stack(templates)
    member_scores = np.concatenate(member_queries) @ gallery.T  # (G, G); diagonal is the true identity
    stranger_queries = _unit(np.stack([table[s][rng.permutation(len(table[s]))[:query]].mean(axis=0) for s in strangers]))
    stranger_scores = stranger_queries @ gallery.T

    correct = member_scores.argmax(axis=1) == np.arange(len(members))
    return member_scores.max(axis=1), correct, stranger_scores.max(axis=1)


def identification_operating_points(member_top, member_correct, stranger_top, fpir_targets) -> dict:
    """Open-set outcomes at thresholds chosen by a cap on the false-positive identification rate.

    Balancing errors (EER) is the wrong target here: for "who typed this?", naming the wrong person
    is worse than declining. So the threshold is the lowest one whose FPIR (strangers who get named
    as someone) stays under the cap, and the enrolled side is split three ways -- named correctly
    (DIR), named as someone else (misidentified), or not named -- since the top score clearing the
    bar says nothing about whether the top NAME is right.
    """
    stranger_top = np.sort(stranger_top)
    out = {}
    for target in fpir_targets:
        # Accept when score >= t; t sits just above the stranger score at the (1 - target) quantile.
        k = min(int(np.ceil(len(stranger_top) * (1 - target))), len(stranger_top) - 1)
        t = float(stranger_top[k]) + 1e-9
        named = member_top >= t
        out[str(target)] = {
            "threshold": round(t, 4),
            "fpir_percent": round(100 * float((stranger_top >= t).mean()), 2),
            "dir_percent": round(100 * float((named & member_correct).mean()), 2),
            "misidentified_percent": round(100 * float((named & ~member_correct).mean()), 2),
            "not_named_percent": round(100 * float((~named).mean()), 2),
        }
    return out


def sweep(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    """One global threshold across everyone: EER, and the FRR you pay at fixed FAR budgets.

    Counts come from searchsorted rather than a threshold x score boolean matrix; at a few
    million impostor scores that matrix is terabytes.
    """
    genuine = np.sort(genuine)
    impostor = np.sort(impostor)
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    # Accept when score >= threshold. FAR = impostors accepted, FRR = genuines rejected.
    far = (len(impostor) - np.searchsorted(impostor, thresholds, side="left")) / len(impostor)
    frr = np.searchsorted(genuine, thresholds, side="left") / len(genuine)

    k = int(np.argmin(np.abs(far - frr)))
    out = {
        "eer_percent": round(100 * float((far[k] + frr[k]) / 2), 3),
        "eer_threshold": round(float(thresholds[k]), 4),
        "genuine_scores": len(genuine),
        "impostor_scores": len(impostor),
    }
    for budget in (0.01, 0.001):
        feasible = np.flatnonzero(far <= budget)
        if len(feasible):
            j = feasible[int(np.argmin(frr[feasible]))]
            out[f"far_{budget}"] = {
                "threshold": round(float(thresholds[j]), 4),
                "frr_percent": round(100 * float(frr[j]), 3),
                "far_percent": round(100 * float(far[j]), 4),
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--subjects", type=int, default=1000)
    parser.add_argument("--outsiders", type=int, default=1000, help="unenrolled people probing identification")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--identify-draws", type=int, default=5, help="random galleries pooled per query count")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = session_embedding_table(load_encoder(args.checkpoint, False, device), load_eval_data(), device)
    results = {"authentication": {}, "identification": {}}

    print("authentication (1:1), one global threshold per cell")
    print(f"{'enroll':>7}{'query':>7}{'people':>8}{'EER%':>8}{'thresh':>8}{'FRR%@FAR=1%':>13}")
    for e in PROMPT_COUNTS:
        for q in PROMPT_COUNTS:
            if e + probes_per_subject(q) * q > 15:
                print(f"{e:>7}{q:>7}   not measurable: needs more than Aalto's 15 sessions per person")
                continue
            genuine, impostor, n = authentication_scores(table, e, q, args.subjects, args.seed)
            r = sweep(genuine, impostor) | {"subjects": n}
            results["authentication"][f"{e},{q}"] = r
            print(f"{e:>7}{q:>7}{n:>8}{r['eer_percent']:>8.2f}{r['eer_threshold']:>8.3f}"
                  f"{r.get('far_0.01', {}).get('frr_percent', float('nan')):>13.2f}")

    fpir_targets = (0.10, 0.05, 0.02, 0.01)
    print(f"\nidentification (open set), gallery of {IDENTIFY_GALLERY} enrolled from {IDENTIFY_ENROLL} prompts, "
          f"{args.outsiders} strangers, {args.identify_draws} draws pooled")
    print(f"{'query':>6}{'rank-1%':>9}{'FPIR cap':>10}{'thresh':>8}{'named ok%':>11}{'wrong name%':>13}{'not named%':>12}")
    for q in PROMPT_COUNTS:
        draws = [identification_scores(table, q, args.outsiders, args.seed + d) for d in range(args.identify_draws)]
        member_top, correct, stranger_top = (np.concatenate(parts) for parts in zip(*draws))
        points = identification_operating_points(member_top, correct, stranger_top, fpir_targets)
        results["identification"][str(q)] = {"rank1_percent": round(100 * float(correct.mean()), 2),
                                             "operating_points": points}
        for target, r in points.items():
            print(f"{q:>6}{100 * correct.mean():>9.1f}{100 * float(target):>9.0f}%{r['threshold']:>8.3f}"
                  f"{r['dir_percent']:>11.1f}{r['misidentified_percent']:>13.1f}{r['not_named_percent']:>12.1f}")

    if args.json_out:
        json.dump(results, open(args.json_out, "w"), indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
