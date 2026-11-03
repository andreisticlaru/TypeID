"""One calibrated score, three decisions: forensic strength of evidence, one-time acceptance, continuous authentication.

A logistic map fitted on half the people turns a distance into a log likelihood ratio (LLR), with equal class weights,
so 0 means the evidence favours neither side. Scored on the other half:
  * Cllr: the cost of the LLRs in bits. 0 is perfect, 1 is what always reporting LLR = 0 costs; above 1 misleads.
  * Tippett: how much genuine and impostor evidence points each way, and how much strongly misleads.
  * SPRT: add the LLR of each 50-key query window, lock out at -log(99) and accept at +log(99). ANGA and ANIA are the
    mean windows until a genuine user is locked out and until an impostor is. Only 10 windows exist per person, so
    anyone not locked out counts as 10: both are lower bounds. Windows of one person are correlated and the sum treats
    them as independent, so the 1% targets are optimistic; the measured rates are the result.
DIR at a fixed FPIR, the open-set identification decision, comes from comparison.paired.

run with: python -m comparison.decisions --split data/split_klicke_dev.json --cache data/preprocessed_klicke_copy_essay \
          --tag klicke_copy_essay --model C1=runs/p6_c_f/final.pt
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from comparison.backend import variants
from comparison.levers import sequence_distances
from comparison.loaders import load_any_encoder
from comparison.paired import KVC_IMPOSTORS, RESULTS_DIR, load_people_embeddings
from data_v2.datasets import load_eval_data

SPRT_BOUND = np.log(99)
TIPPETT_POINTS = (-2.0, -1.0, 0.0, 1.0, 2.0)  # log10 LR


def fit_calibration(genuine: np.ndarray, impostor: np.ndarray, steps: int = 50) -> tuple[float, float]:
    """(a, b) of LLR = a * distance + b, by Newton on the class-balanced logistic loss."""
    x = np.concatenate([genuine, impostor])
    y = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    w = np.concatenate([np.full(len(genuine), .5 / len(genuine)), np.full(len(impostor), .5 / len(impostor))])
    mean, std = x.mean(), x.std()
    z = np.stack([(x - mean) / std, np.ones(len(x))], axis=1)
    theta = np.zeros(2)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-z @ theta))
        hessian = (z * (w * p * (1 - p))[:, None]).T @ z + 1e-9 * np.eye(2)
        theta -= np.linalg.solve(hessian, z.T @ (w * (p - y)))
    return theta[0] / std, theta[1] - theta[0] * mean / std


def cllr(genuine: np.ndarray, impostor: np.ndarray) -> float:
    return float((np.logaddexp(0, -genuine).mean() + np.logaddexp(0, impostor).mean()) / (2 * np.log(2)))


def sprt(llrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For (..., Q) window LLRs: windows until lock-out (Q if never) and whether acceptance came first or nothing did."""
    total = np.cumsum(llrs, axis=-1)
    q = llrs.shape[-1]
    first = lambda hit: np.where(hit.any(axis=-1), hit.argmax(axis=-1) + 1, q + 1)  # noqa: E731
    lock, accept = first(total <= -SPRT_BOUND), first(total >= SPRT_BOUND)
    locked = lock < accept
    return np.where(locked, lock, q), accept < lock


def draw_scores(emb: torch.Tensor, seed: int, device) -> tuple[np.ndarray, np.ndarray]:
    """Per-window distances: genuine (n, Q), impostor (n, I, Q), I random other people."""
    n = len(emb)
    d = np.random.default_rng(seed).random((n, n))
    np.fill_diagonal(d, np.inf)
    others = np.argsort(d, axis=1)[:, :KVC_IMPOSTORS]
    return sequence_distances(emb, np.arange(n), others, device)


def report_one(fit: torch.Tensor, emb: torch.Tensor, device, B: int = 500) -> dict:
    a, b = fit_calibration(*(x.ravel() for x in draw_scores(fit, 0, device)))
    genuine, impostor = draw_scores(emb, 1, device)
    g, i = a * genuine + b, a * impostor + b
    n = len(g)
    rng = np.random.default_rng(2)
    boot = [cllr(g[idx].ravel(), i[idx].ravel()) for idx in (rng.integers(0, n, n) for _ in range(B))]
    until_g, accepted_g = sprt(g)
    until_i, accepted_i = sprt(i)
    log10g, log10i = g.ravel() / np.log(10), i.ravel() / np.log(10)
    return {"cllr": round(cllr(g.ravel(), i.ravel()), 3), "cllr_ci95": np.percentile(boot, [2.5, 97.5]).round(3).tolist(),
            "tippett_log10lr": list(TIPPETT_POINTS),
            "genuine_above": [round(100 * float((log10g > t).mean()), 2) for t in TIPPETT_POINTS],
            "impostor_above": [round(100 * float((log10i > t).mean()), 2) for t in TIPPETT_POINTS],
            "misleading_strong_genuine": round(100 * float((log10g < -1).mean()), 2),
            "misleading_strong_impostor": round(100 * float((log10i > 1).mean()), 2),
            "anga_lower_bound": round(float(until_g.mean()), 2), "ania_lower_bound": round(float(until_i.mean()), 2),
            "genuine_accepted": round(100 * float(accepted_g.mean()), 1), "impostor_accepted": round(100 * float(accepted_i.mean()), 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--split", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--tag", required=True, help="result file is decisions_<tag>.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_eval_data(args.split, args.cache)
    out = {"split": args.split, "cache": args.cache, "models": {}}
    for spec in args.model:
        name, path = spec.split("=", 1)
        emb, _ = load_people_embeddings(load_any_encoder(path, device), data, device)
        order = torch.from_numpy(np.random.default_rng(1).permutation(len(emb)))
        fit, scored = emb[order[:len(emb) // 2]], emb[order[len(emb) // 2:]]
        fit_by, scored_by = variants(fit, fit), variants(fit, scored)  # the back-end is learned on fit people only
        out["models"][name] = {v: report_one(fit_by[v], scored_by[v], device) for v in ("cosine", "center+wccn")}
        for v, r in out["models"][name].items():
            print(f"{name:<4} {v:<12} Cllr {r['cllr']:.3f} {r['cllr_ci95']}  mislead>10:1 gen {r['misleading_strong_genuine']:.1f}% imp "
                  f"{r['misleading_strong_impostor']:.1f}%  ANGA>={r['anga_lower_bound']} ANIA>={r['ania_lower_bound']}  "
                  f"accepted gen {r['genuine_accepted']}% imp {r['impostor_accepted']}%", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS_DIR / f"decisions_{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
