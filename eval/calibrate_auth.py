"""Calibrate a single GLOBAL decision threshold for 1:1 authentication (verification).

Why this exists separately from eval/typenet_protocol.py: that script reports TypeNet's
per-subject EER, where the operating point is chosen from each person's own genuine and
impostor scores. That measures separability and is the right thing for a like-for-like
comparison with the paper, but it is an oracle -- a deployed authenticator has to commit to
ONE threshold in advance, before it has ever seen the claimant's score distribution. The
per-subject number is therefore optimistic as a deployment figure, and can't be lifted into
the backend.

This script fixes one threshold across all held-out subjects and reports what it actually
costs, scoring exactly the way POST /authenticate does:

  template = mean of the per-session embeddings of E enrollment sessions   (enroll.py pooling)
  query    = pooled_embedding of one probe session                         (embedding.py)
  score    = cosine(query, template)                                       (identify.py)

Genuine = query vs own template. Impostor = query vs every other subject's template.
Subjects are held-out (split.json eval_subjects), so no identity here was seen in training.

run with: python -m eval.calibrate_auth --checkpoint model/encoder_hard.pt
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from eval.rank_n import embed_rows, load_encoder, load_eval_data, pooled_embedding
from model.train import CHECKPOINT_PATH

PROBE_SESSIONS = 3  # per subject, held out from the enrollment sessions


def session_embedding_table(model, data, device, min_sessions):
    """{subject: (n_sessions, D)} -- one pooled embedding per session, backend-identical pooling."""
    windows, mask, eval_rows, rows_by_session = data
    all_embeddings = embed_rows(model, windows, mask, eval_rows, device)
    row_index = {row: i for i, row in enumerate(eval_rows)}

    by_subject: dict[str, list] = {}
    for (subject, session), rows in rows_by_session.items():
        by_subject.setdefault(subject, []).append(
            pooled_embedding(all_embeddings[[row_index[r] for r in rows]])
        )
    return {s: np.stack(v) for s, v in by_subject.items() if len(v) >= min_sessions}


def genuine_and_impostor(table, enroll_sessions, rng):
    """Pooled cosine scores under one global threshold: (genuine, impostor) score vectors."""
    subjects = sorted(table)
    templates, probes = [], []
    for s in subjects:
        sessions = table[s]
        order = rng.permutation(len(sessions))
        enrolled = sessions[order[:enroll_sessions]]
        templates.append(enrolled.mean(axis=0))  # enroll.py pools per-session embeddings by mean
        probes.append(sessions[order[enroll_sessions:enroll_sessions + PROBE_SESSIONS]])

    # Cosine, computed the way identify.py does (explicit norms, templates are not re-normalized).
    t = np.stack(templates)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    p = np.stack(probes)  # (n, PROBE_SESSIONS, D); probe embeddings are already unit norm
    scores = np.einsum("npd,md->npm", p, t)  # [i, k, j] = subject i's probe k vs subject j's template

    n = len(subjects)
    own = np.arange(n)
    genuine = scores[own, :, own].ravel()
    impostor = scores[~np.eye(n, dtype=bool)[:, None, :].repeat(PROBE_SESSIONS, axis=1)].ravel()
    return genuine, impostor


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
    parser.add_argument("--enroll-sessions", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--subjects", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_eval_data()
    max_enroll = max(args.enroll_sessions)
    table = session_embedding_table(
        load_encoder(args.checkpoint, False, device), data, device, max_enroll + PROBE_SESSIONS
    )

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(sorted(table), size=min(args.subjects, len(table)), replace=False)
    table = {s: table[s] for s in chosen}

    print(f"global-threshold authentication, {len(table)} held-out subjects, {PROBE_SESSIONS} probes each")
    print(f"{'enroll':>7}{'EER%':>9}{'thresh':>9}{'FRR%@FAR=1%':>14}{'FRR%@FAR=0.1%':>15}")
    results = {}
    for e in args.enroll_sessions:
        r = sweep(*genuine_and_impostor(table, e, np.random.default_rng(args.seed)))
        results[e] = r
        one = r.get("far_0.01", {})
        tenth = r.get("far_0.001", {})
        print(
            f"{e:>7}{r['eer_percent']:>9.2f}{r['eer_threshold']:>9.3f}"
            f"{one.get('frr_percent', float('nan')):>14.2f}{tenth.get('frr_percent', float('nan')):>15.2f}"
        )

    if args.json_out:
        json.dump({"subjects": len(table), "results": results}, open(args.json_out, "w"), indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
