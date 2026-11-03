"""Back-ends on frozen embeddings: can a linear map learned on other people remove a condition change (task, session)?

Borrowed from speaker recognition, which handles channel changes after the embedding instead of retraining it.
Each is fitted on layouts with two conditions in one person's 15 sequences (first 5 = enrollment, last 10 = queries):
  center: subtract each condition's mean embedding. Enough if the condition moves everyone the same way.
  wccn:   whiten by the within-person covariance, so directions that vary within one person count less. Enough if
          the move differs per person but stays in a few directions.
  center+wccn: both.
Each is followed by length-normalization. Scored with comparison.levers' KVC metric, plain and with s-norm, on the
same people and impostor draws, so all differences are paired.

Fit people are never scored: without --fit-cache the eval people are cut in two and the first half is the fit set;
with it, the fit set is another dataset's people and every eval person is scored.

run with: python -m comparison.backend --split data/split_klicke_dev.json --cache data/preprocessed_klicke_copy_essay \
          --tag klicke_copy_essay --model C1=runs/p6_c_f/final.pt
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from comparison.levers import cohort_stats, kvc_dict, pool, sequence_distances, snorm
from comparison.loaders import load_any_encoder
from comparison.paired import KVC_ENROLL, KVC_IMPOSTORS, RESULTS_DIR, kvc_global, load_people_embeddings
from data_v2.datasets import load_eval_data

SHRINK = 0.05  # of the mean eigenvalue: a few dozen fit people cannot pin down 128 directions


def condition_means(fit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean embedding of the enrollment and of the query condition."""
    return fit[:, :KVC_ENROLL].mean(dim=(0, 1)), fit[:, KVC_ENROLL:].mean(dim=(0, 1))


def center(emb: torch.Tensor, means: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return torch.cat([emb[:, :KVC_ENROLL] - means[0], emb[:, KVC_ENROLL:] - means[1]], dim=1)


def wccn_map(fit: torch.Tensor) -> torch.Tensor:
    """W with W'W = inverse within-person covariance, so Euclidean distance after x @ W ignores within-person spread."""
    residual = (fit - fit.mean(dim=1, keepdim=True)).reshape(-1, fit.shape[-1]).double()
    cov = residual.T @ residual / (len(residual) - len(fit))
    cov += SHRINK * cov.diagonal().mean() * torch.eye(len(cov), dtype=cov.dtype)
    return torch.linalg.cholesky(torch.linalg.inv(cov)).float()


def variants(fit: torch.Tensor, emb: torch.Tensor) -> dict[str, torch.Tensor]:
    means = condition_means(fit)
    centered_fit, centered = center(fit, means), center(emb, means)
    overall = fit.mean(dim=(0, 1))
    return {"cosine": emb,
            "center": F.normalize(centered, dim=-1),
            "wccn": F.normalize((emb - overall) @ wccn_map(fit), dim=-1),
            "center+wccn": F.normalize(centered @ wccn_map(centered_fit), dim=-1)}


def score_variants(by_variant: dict[str, torch.Tensor], repetitions: int, cohort_size: int, B: int, device) -> dict:
    """KVC EER / FNMR@1% per variant, plain and s-normed, with bootstrap differences to plain cosine."""
    n_all = len(next(iter(by_variant.values())))
    rng = np.random.default_rng(0)
    order = rng.permutation(n_all)
    cohort_size = min(cohort_size, n_all // 5)
    cohort, scored = np.sort(order[:cohort_size]), np.sort(order[cohort_size:])
    n, pos = len(scored), np.arange(len(scored))
    draws = []
    for _ in range(repetitions):
        d = rng.random((n, n))
        np.fill_diagonal(d, np.inf)
        draws.append((np.argsort(d, axis=1)[:, :KVC_IMPOSTORS], rng.integers(0, 10 ** 6, size=(n, KVC_IMPOSTORS))))
    boot = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)])

    points, dists = {}, {}
    for name, emb in by_variant.items():
        stats = cohort_stats(emb, scored, cohort, device)
        collected = {name: [], f"{name} snorm": []}
        for others_pos, pick in draws:
            g, i = sequence_distances(emb, scored, scored[others_pos], device)
            collected[name].append(kvc_dict(*pool(g, i, 1, pick), pos)["kvc"])
            collected[f"{name} snorm"].append(kvc_dict(*pool(*snorm(g, i, others_pos, stats), 1, pick), pos)["kvc"])
        for label, parts in collected.items():
            scores = {"kvc": {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}}
            points[label], dists[label] = kvc_global(scores, np.ones((1, n), dtype=np.int64))[0], kvc_global(scores, boot)
    report = {"scored": n, "cohort": cohort_size, "variants": {}}
    for label, point in points.items():
        delta = dists[label][:, 0] - dists["cosine"][:, 0]
        report["variants"][label] = {
            "kvc_eer": round(float(point[0]), 2), "kvc_fnmr1": round(float(point[1]), 2),
            "kvc_eer_ci95": np.percentile(dists[label][:, 0], [2.5, 97.5]).round(2).tolist(),
            "delta_vs_cosine": round(float(point[0] - points["cosine"][0]), 2),
            "delta_ci95": np.percentile(delta, [2.5, 97.5]).round(2).tolist()}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--split", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--fit-split", default=None)
    parser.add_argument("--fit-cache", default=None)
    parser.add_argument("--tag", required=True, help="result file is backend_<tag>.json")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cohort", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_eval_data(args.split, args.cache)
    report = {"split": args.split, "cache": args.cache, "fit_split": args.fit_split, "fit_cache": args.fit_cache,
              "models": {}}
    for spec in args.model:
        name, path = spec.split("=", 1)
        model = load_any_encoder(path, device)
        emb, _ = load_people_embeddings(model, data, device)
        if args.fit_cache:
            fit, _ = load_people_embeddings(model, load_eval_data(args.fit_split, args.fit_cache), device)
        else:
            order = torch.from_numpy(np.random.default_rng(1).permutation(len(emb)))
            fit, emb = emb[order[:len(emb) // 2]], emb[order[len(emb) // 2:]]
        row = score_variants(variants(fit, emb), args.repetitions, args.cohort, args.bootstrap, device)
        row["fit_people"] = len(fit)
        report["models"][name] = row
        print(f"\n{name}: fit on {len(fit)} people, scored {row['scored']} (+{row['cohort']} cohort)", flush=True)
        for label, v in row["variants"].items():
            print(f"{label:<20} EER {v['kvc_eer']:6.2f} [{v['kvc_eer_ci95'][0]:.2f},{v['kvc_eer_ci95'][1]:.2f}]  FNMR@1% {v['kvc_fnmr1']:6.2f}"
                  f"  vs cosine {v['delta_vs_cosine']:+.2f} [{v['delta_ci95'][0]:+.2f},{v['delta_ci95'][1]:+.2f}]", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    json.dump(report, open(RESULTS_DIR / f"backend_{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
