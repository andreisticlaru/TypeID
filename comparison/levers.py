"""Cheap levers on existing encoders, scored with the KVC-style metric: more typing per query, ensembles, s-norm.

Same layout as comparison.paired's KVC metric: per person, sequences 1-5 enroll, 6-15 are queries; a score is
the mean distance from the 5 enrollment sequences; ONE global threshold (global EER, FNMR at 1% FMR).

  * pool k: a query is k consecutive query sequences (score = mean over 5 x k distances): "type k times more".
    k=1 is the plain metric. 10 // k genuine queries per person.
  * ensemble: each model's distances divided by its median impostor distance, then averaged over models.
  * s-norm: each score standardized against a cohort, from both sides and averaged:
      s = ((d - mean_e) / std_e + (d - mean_q) / std_q) / 2
    mean_e/std_e: the enrolled person's scores against every cohort query sequence; mean_q/std_q: the query's
    scores against every cohort enrollment. Cohort people are never scored and never used as impostors.

Impostors: per person and repetition, 20 random other non-cohort people, one random query (group) each; the
draws are shared by every model and variant, and the bootstrap resamples people with shared draws, so
differences between variants are paired.

run with: python -m comparison.levers --split data/split_unseen_both.json --cache data/preprocessed_typenet \
          --tag unseen_nf --model C1=runs/p6_c_f/final.pt --model C1_s1=runs/p6_c_f_s1/final.pt
The first --model is the reference for the ensemble rows.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from comparison.loaders import load_any_encoder
from comparison.paired import KVC_ENROLL, KVC_IMPOSTORS, RESULTS_DIR, kvc_global, load_people_embeddings
from data_v2.datasets import load_eval_data

POOLS = (1, 2, 3)


def sequence_distances(emb: torch.Tensor, scored: np.ndarray, others: np.ndarray, device) -> tuple:
    """Per query SEQUENCE, mean distance to the 5 enrollment sequences.

    emb (N, 15, D); scored (n,) person indices; others (n, I) impostor person indices.
    Returns genuine (n, Q) and impostor (n, I, Q), Q = 15 - KVC_ENROLL.
    """
    out_g, out_i = [], []
    for s in range(0, len(scored), 1000):
        idx = torch.from_numpy(scored[s:s + 1000])
        enroll = emb[idx, :KVC_ENROLL].to(device)  # (b, 5, D)
        own = emb[idx, KVC_ENROLL:].to(device)  # (b, Q, D)
        out_g.append(torch.cdist(enroll, own).mean(dim=1).cpu())
        theirs = emb[torch.from_numpy(others[s:s + 1000]), KVC_ENROLL:].to(device)  # (b, I, Q, D)
        b, i, q, d = theirs.shape
        out_i.append(torch.cdist(enroll, theirs.reshape(b, i * q, d)).mean(dim=1).reshape(b, i, q).cpu())
    return torch.cat(out_g).numpy(), torch.cat(out_i).numpy()


def cohort_stats(emb: torch.Tensor, scored: np.ndarray, cohort: np.ndarray, device) -> tuple:
    """s-norm statistics at sequence level: (mean_e, std_e) per scored person, (mean_q, std_q) per (person, query)."""
    c_enroll = emb[torch.from_numpy(cohort), :KVC_ENROLL].to(device)  # (C, 5, D)
    c_query = emb[torch.from_numpy(cohort), KVC_ENROLL:].reshape(-1, emb.shape[-1]).to(device)  # (C*Q, D)
    mean_e, std_e, mean_q, std_q = [], [], [], []
    for s in range(0, len(scored), 1000):
        idx = torch.from_numpy(scored[s:s + 1000])
        enroll = emb[idx, :KVC_ENROLL].to(device)  # (b, 5, D)
        e = torch.cdist(enroll, c_query[None].expand(len(idx), -1, -1)).mean(dim=1)  # (b, C*Q)
        mean_e.append(e.mean(dim=1).cpu()); std_e.append(e.std(dim=1).cpu())
        query = emb[idx, KVC_ENROLL:].to(device)  # (b, Q, D)
        b, q, d = query.shape
        qd = torch.cdist(query.reshape(b * q, d), c_enroll.reshape(-1, d)).reshape(b, q, len(cohort), KVC_ENROLL)
        qd = qd.mean(dim=3)  # (b, Q, C): each cohort person's mean over their 5 enrollment sequences
        mean_q.append(qd.mean(dim=2).cpu()); std_q.append(qd.std(dim=2).cpu())
    return tuple(torch.cat(x).numpy() for x in (mean_e, std_e, mean_q, std_q))


def snorm(genuine, impostor, others_pos, stats):
    """Apply s-norm to sequence-level genuine (n, Q) and impostor (n, I, Q) distances."""
    mean_e, std_e, mean_q, std_q = stats
    g = ((genuine - mean_e[:, None]) / std_e[:, None] + (genuine - mean_q) / std_q) / 2
    me, se = mean_e[:, None, None], std_e[:, None, None]
    mq, sq = mean_q[others_pos], std_q[others_pos]  # (n, I, Q): the impostor query's own cohort stats
    i = ((impostor - me) / se + (impostor - mq) / sq) / 2
    return g, i


def pool(genuine, impostor, k: int, pick: np.ndarray):
    """Group consecutive query sequences by k: genuine (n, Q//k) means; impostor (n, I), group pick[n, I]."""
    groups = genuine.shape[1] // k
    g = genuine[:, :groups * k].reshape(len(genuine), groups, k).mean(axis=2)
    i = impostor[:, :, :groups * k].reshape(*impostor.shape[:2], groups, k).mean(axis=3)
    return g, np.take_along_axis(i, (pick % groups)[:, :, None], axis=2)[:, :, 0]


def kvc_dict(genuine, impostor, people_pos):
    return {"kvc": {"genuine": genuine.ravel(), "impostor": impostor.ravel(),
                    "genuine_person": np.repeat(people_pos, genuine.shape[1]),
                    "impostor_person": np.repeat(people_pos, impostor.shape[1])}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--split", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--tag", required=True, help="result file is levers_<tag>.json")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cohort", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_eval_data(args.split, args.cache)
    embeddings, people = {}, None
    for spec in args.model:
        name, path = spec.split("=", 1)
        embeddings[name], these = load_people_embeddings(load_any_encoder(path, device), data, device)
        assert people is None or these == people, "models must be scored on the same people"
        people = these
    N = len(people)
    rng = np.random.default_rng(0)
    order = rng.permutation(N)
    cohort_size = min(args.cohort, N // 5)
    cohort, scored = np.sort(order[:cohort_size]), np.sort(order[cohort_size:])
    n = len(scored)
    pos = np.arange(n)
    print(f"{N} people: {n} scored, {cohort_size} cohort; {args.repetitions} repetitions", flush=True)

    # per repetition: impostor people (positions within `scored`) and which query (group) of theirs is used
    draws = []
    for r in range(args.repetitions):
        d = rng.random((n, n))
        np.fill_diagonal(d, np.inf)
        draws.append((np.argsort(d, axis=1)[:, :KVC_IMPOSTORS], rng.integers(0, 10 ** 6, size=(n, KVC_IMPOSTORS))))

    # sequence-level distances per model and repetition (and s-normed copies)
    raw, normed = {}, {}
    for name, emb in embeddings.items():
        stats = cohort_stats(emb, scored, cohort, device)
        raw[name], normed[name] = [], []
        for others_pos, _ in draws:
            g, i = sequence_distances(emb, scored, scored[others_pos], device)
            raw[name].append((g, i))
            normed[name].append(snorm(g, i, others_pos, stats))
        print(f"distances for {name}", flush=True)
    scale = {name: np.median(raw[name][0][1]) for name in raw}  # median impostor distance, for the ensemble

    def ensemble(source, names, normalize):
        return [tuple(sum(source[m][r][j] / (scale[m] if normalize else 1) for m in names) / len(names)
                      for j in (0, 1)) for r in range(args.repetitions)]

    variants = {}
    for name in raw:
        for k in POOLS:
            variants[f"{name} k={k}"] = (raw[name], k)
        variants[f"{name} snorm"] = (normed[name], 1)
    if len(raw) > 1:
        names = list(raw)
        for k in POOLS:
            variants[f"ensemble k={k}"] = (ensemble(raw, names, True), k)
        variants["ensemble snorm"] = (ensemble(normed, names, False), 1)

    counts = np.ones((1, n), dtype=np.int64)
    boot = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(args.bootstrap)])
    results, dists = {}, {}
    for label, (per_rep, k) in variants.items():
        gs, is_, gp, ip = [], [], [], []
        for (g, i), (_, pick) in zip(per_rep, draws):
            pg, pi = pool(g, i, k, pick)
            d = kvc_dict(pg, pi, pos)["kvc"]
            gs.append(d["genuine"]); is_.append(d["impostor"]); gp.append(d["genuine_person"]); ip.append(d["impostor_person"])
        scores = {"kvc": {"genuine": np.concatenate(gs), "impostor": np.concatenate(is_),
                          "genuine_person": np.concatenate(gp), "impostor_person": np.concatenate(ip)}}
        point = kvc_global(scores, counts)[0]
        dists[label] = kvc_global(scores, boot)
        results[label] = {"kvc_eer": round(float(point[0]), 2), "kvc_fnmr1": round(float(point[1]), 2),
                          "kvc_eer_ci95": np.percentile(dists[label][:, 0], [2.5, 97.5]).round(2).tolist()}
        print(f"{label:<22} KVC EER {point[0]:5.2f}  FNMR@1% {point[1]:6.2f}", flush=True)

    # paired differences: every variant against each model's plain k=1
    for label in results:
        ref = f"{label.split(' ')[0]} k=1" if not label.startswith("ensemble") else f"{list(raw)[0]} k=1"
        if label != ref:
            delta = dists[label][:, 0] - dists[ref][:, 0]
            results[label]["vs"] = ref
            results[label]["delta_kvc_eer"] = round(results[label]["kvc_eer"] - results[ref]["kvc_eer"], 2)
            results[label]["delta_ci95"] = np.percentile(delta, [2.5, 97.5]).round(2).tolist()

    RESULTS_DIR.mkdir(exist_ok=True)
    report = {"people": N, "scored": n, "cohort": cohort_size, "repetitions": args.repetitions,
              "bootstrap": args.bootstrap, "split": args.split, "cache": args.cache,
              "checkpoints": dict(s.split("=", 1) for s in args.model), "results": results}
    json.dump(report, open(RESULTS_DIR / f"levers_{args.tag}.json", "w"), indent=2)
    print("\nvariant                 KVC EER  [95% CI]        vs          delta [95% CI]")
    for label, r in results.items():
        extra = f"  {r['vs']:<11} {r['delta_kvc_eer']:+.2f} [{r['delta_ci95'][0]:+.2f},{r['delta_ci95'][1]:+.2f}]" if "vs" in r else ""
        print(f"{label:<22} {r['kvc_eer']:6.2f} [{r['kvc_eer_ci95'][0]:.2f},{r['kvc_eer_ci95'][1]:.2f}]{extra}")


if __name__ == "__main__":
    main()
