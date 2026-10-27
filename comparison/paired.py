"""Paired, person-level comparison of encoders under TypeNet's protocol.

comparison/evaluate.py reports mean +- std over 10 random draws of 1,000 people. The draws overlap (10 draws
cover ~6.5k of the 10k unseen people) and the std measures draw-to-draw noise, not "is model A better than B".
This script scores the SAME people with every model and tests the per-person difference:

  * People are shuffled and cut into disjoint backgrounds of 1,000 (a short last chunk is topped up with
    people from earlier chunks, which are not scored). Every person is scored once per repetition.
  * Closed-set Rank-1/5 (TypeNet's rule: 10 gallery + 5 query sequences, mean pairwise distance).
  * Per-person authentication EER at G = 1/5/10 (same rule as eval.typenet_protocol.authentication_eer).
  * Open-set: the first half of each background is enrolled, the second half are strangers. Detect-and-
    identify rate (DIR) = enrolled queries named correctly AND accepted, at the threshold where
    FAR (strangers named) is 1% / 5%. This is the "no match" threshold the live app needs.
  * KVC-style verification (Keystroke Verification Challenge, desktop): first 5 sequences enroll, last 10 query;
    10 genuine + 20 impostor scores per person (one query each from 20 others in the background), each the mean
    distance to the 5 enrollment sequences, and ONE threshold for everyone: global EER and FNMR at 1% FMR.
    KVC draws half its impostors from the same age/gender group; we have no demographics, so ours are random.
  * Cluster bootstrap over PEOPLE (the independent unit) for every number and for every difference to the
    reference model. All models share the same resamples, so differences are paired.

run with: python -m comparison.paired --split data/split_unseen_both.json --cache data/preprocessed_typenet \
          --tag unseen_nf --model v1=model/encoder_hard.pt --model v2=runs/p3_v2long_hard/final.pt
The first --model is the reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from comparison.loaders import load_any_encoder
from data_v2.datasets import load_eval_data
from eval.typenet_protocol import GALLERY_MAX, QUERIES, SEQUENCES, mean_pairwise_distance, session_embeddings

RESULTS_DIR = Path(__file__).parent / "results"
EER_G = (1, 5, 10)
FAR_POINTS = (0.01, 0.05)
KVC_ENROLL, KVC_IMPOSTORS = 5, 20  # the other SEQUENCES - KVC_ENROLL = 10 sequences are the queries
NON_INFERIORITY_MARGIN = 1.0  # Rank-1 points; the plan's 2-point rule is for adopting a change, this is for "not worse than v1"


def partitions(n: int, seed: int, size: int) -> list[tuple[np.ndarray, int]]:
    """Disjoint backgrounds of `size` people; (indices, number of scored people at the front)."""
    size = min(size, n)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    chunks = []
    for start in range(0, n, size):
        chunk = order[start:start + size]
        real = len(chunk)
        if real < size:  # top up with already-scored people so every background has `size` people
            chunk = np.concatenate([chunk, rng.choice(order[:start], size - real, replace=False)])
        chunks.append((chunk, real))
    return chunks


def batched_eer(genuine: torch.Tensor, impostor: torch.Tensor) -> torch.Tensor:
    """Per-row EER, identical to eval.typenet_protocol.subject_eer: (n, k) genuine, (n, m) impostor -> (n,)."""
    thresholds = torch.cat([genuine, impostor], dim=1).sort(dim=1).values.contiguous()
    g_sorted, i_sorted = genuine.sort(dim=1).values.contiguous(), impostor.sort(dim=1).values.contiguous()
    frr = 1 - torch.searchsorted(g_sorted, thresholds, right=True) / genuine.shape[1]  # genuine > t
    far = torch.searchsorted(i_sorted, thresholds, right=True) / impostor.shape[1]  # impostor <= t
    k = (far - frr).abs().argmin(dim=1, keepdim=True)
    return ((far.gather(1, k) + frr.gather(1, k)) / 2).squeeze(1)


def person_eers(background: torch.Tensor, gallery_size: int, rng) -> np.ndarray:
    """Per-person EER (%) with `gallery_size` enrollment sequences; same sampling as authentication_eer."""
    n, device = background.shape[0], background.device
    gallery, query = background[:, :gallery_size], background[:, GALLERY_MAX:]
    genuine = torch.cdist(gallery, query).mean(dim=1)  # (n, QUERIES)
    pick = torch.from_numpy(rng.integers(0, QUERIES, size=n)).to(device)  # one impostor sample per person
    one_query = query[torch.arange(n, device=device), pick]
    impostor = torch.cdist(gallery.reshape(n * gallery_size, -1), one_query).reshape(n, gallery_size, n).mean(dim=1)
    impostor = impostor[~torch.eye(n, dtype=torch.bool, device=device)].reshape(n, n - 1)
    return 100 * batched_eer(genuine, impostor).cpu().numpy()


def kvc_scores(background: torch.Tensor, rng) -> tuple[np.ndarray, np.ndarray]:
    """(n, 10) genuine and (n, KVC_IMPOSTORS) impostor distances, each against a person's 5 enrollment sequences."""
    n = background.shape[0]
    enroll, query = background[:, :KVC_ENROLL], background[:, KVC_ENROLL:]
    genuine = torch.cdist(enroll, query).mean(dim=1)
    draw = rng.random((n, n))
    np.fill_diagonal(draw, np.inf)  # never yourself
    others = torch.from_numpy(np.argsort(draw, axis=1)[:, :KVC_IMPOSTORS])
    pick = torch.from_numpy(rng.integers(0, query.shape[1], size=(n, KVC_IMPOSTORS)))
    impostor = torch.cdist(enroll, query[others, pick]).mean(dim=1)
    return genuine.cpu().numpy(), impostor.cpu().numpy()


def score_model(embeddings: torch.Tensor, repetitions: int, size: int, device) -> dict:
    """Per-person outcomes from an (N, 15, D) tensor of every person's first 15 sequence embeddings."""
    n = embeddings.shape[0]
    nan = lambda: np.full((repetitions, n), np.nan)  # noqa: E731
    out = {"hit1": nan(), "hit5": nan(), "eer": {g: nan() for g in EER_G}}
    enrolled = {"person": [], "correct": [], "dist": []}
    strangers = {"person": [], "dist": []}
    kvc = {"genuine_person": [], "genuine": [], "impostor_person": [], "impostor": []}
    for r in range(repetitions):
        rng = np.random.default_rng(1000 + r)
        kvc_rng = np.random.default_rng(2000 + r)  # separate, so the older metrics' draws stay as they were
        for chunk, real in partitions(n, r, size):
            background = embeddings[torch.from_numpy(chunk)].to(device)
            dist = mean_pairwise_distance(background[:, :GALLERY_MAX], background[:, GALLERY_MAX:])  # [gallery i, query j]
            ranks = (1 + (dist < dist.diag()[None, :]).sum(dim=0)).cpu().numpy()
            people = chunk[:real]
            out["hit1"][r, people] = ranks[:real] <= 1
            out["hit5"][r, people] = ranks[:real] <= 5
            for g in EER_G:
                out["eer"][g][r, people] = person_eers(background, g, rng)[:real]
            half = len(chunk) // 2  # open set: enroll the first half, the second half are strangers
            nearest, who = dist[:half].min(dim=0)
            nearest, who = nearest.cpu().numpy(), who.cpu().numpy()
            e = np.arange(min(half, real))
            enrolled["person"].append(chunk[e]); enrolled["correct"].append(who[e] == e); enrolled["dist"].append(nearest[e])
            s = np.arange(half, real)
            strangers["person"].append(chunk[s]); strangers["dist"].append(nearest[s])
            genuine, impostor = kvc_scores(background, kvc_rng)
            kvc["genuine_person"].append(np.repeat(people, genuine.shape[1])); kvc["genuine"].append(genuine[:real].ravel())
            kvc["impostor_person"].append(np.repeat(people, impostor.shape[1])); kvc["impostor"].append(impostor[:real].ravel())
    out["enrolled"] = {k: np.concatenate(v) for k, v in enrolled.items()}
    out["strangers"] = {k: np.concatenate(v) for k, v in strangers.items()}
    out["kvc"] = {k: np.concatenate(v) for k, v in kvc.items()}
    return out


def per_person(scores: dict) -> dict[str, np.ndarray]:
    """Collapse repetitions: one value per person for each closed-set / authentication metric."""
    values = {"rank1": 100 * np.nanmean(scores["hit1"], axis=0), "rank5": 100 * np.nanmean(scores["hit5"], axis=0)}
    values.update({f"eer_g{g}": np.nanmean(scores["eer"][g], axis=0) for g in EER_G})
    return values


def dir_at_far(scores: dict, counts: np.ndarray, far: float) -> np.ndarray:
    """Detect-and-identify rate (%) at the threshold that names `far` of strangers, per bootstrap row of `counts`."""
    st, en = scores["strangers"], scores["enrolled"]
    order = np.argsort(st["dist"])
    st_dist, st_w = st["dist"][order], counts[:, st["person"][order]]
    accepted = np.cumsum(st_w, axis=1) <= far * st_w.sum(axis=1, keepdims=True)
    k = accepted.sum(axis=1)  # strangers accepted at the largest threshold that keeps FAR <= far
    threshold = np.where(k > 0, st_dist[np.maximum(k - 1, 0)], -np.inf)
    en_w = counts[:, en["person"]]
    hit = en["correct"][None, :] & (en["dist"][None, :] <= threshold[:, None])
    return 100 * (en_w * hit).sum(axis=1) / en_w.sum(axis=1)


def kvc_global(scores: dict, counts: np.ndarray) -> np.ndarray:
    """(rows, 2): global EER (%) and FNMR (%) at 1% FMR with one threshold for all people, per row of `counts`.
    A bootstrap row weights every score by how often its person was drawn; same EER rule as batched_eer."""
    k = scores["kvc"]
    dist = np.concatenate([k["genuine"], k["impostor"]])
    person = np.concatenate([k["genuine_person"], k["impostor_person"]])
    is_genuine = np.arange(len(dist)) < len(k["genuine"])
    order = np.argsort(dist, kind="stable")
    person, is_genuine = person[order], is_genuine[order]
    out = np.empty((len(counts), 2))
    for row, c in enumerate(counts):
        w = c[person].astype(float)
        accepted_genuine = np.cumsum(np.where(is_genuine, w, 0))
        accepted_impostor = np.cumsum(np.where(is_genuine, 0, w))
        frr = 1 - accepted_genuine / accepted_genuine[-1]  # genuine above the threshold
        far = accepted_impostor / accepted_impostor[-1]  # impostors at or below it
        i = np.abs(far - frr).argmin()
        ok = np.flatnonzero(far <= 0.01)
        out[row] = 100 * (far[i] + frr[i]) / 2, 100 * (frr[ok[-1]] if len(ok) else 1.0)
    return out


def bootstrap(models: dict[str, dict], n_people: int, B: int, seed: int = 0, chunk: int = 100) -> dict:
    """Metric -> model -> (B,) bootstrap distribution, resampling people with shared draws across models."""
    values = {name: per_person(s) for name, s in models.items()}
    metrics = list(next(iter(values.values())))
    dist = {m: {name: np.empty(B) for name in models} for m in metrics}
    dist.update({f"dir_far{int(100 * f)}": {name: np.empty(B) for name in models} for f in FAR_POINTS})
    dist.update({m: {name: np.empty(B) for name in models} for m in ("kvc_eer", "kvc_fnmr1")})
    rng = np.random.default_rng(seed)
    for start in range(0, B, chunk):
        idx = rng.integers(0, n_people, size=(min(chunk, B - start), n_people))
        counts = np.stack([np.bincount(row, minlength=n_people) for row in idx])
        for name in models:
            for m in metrics:
                dist[m][name][start:start + len(idx)] = counts @ values[name][m] / n_people
            for f in FAR_POINTS:
                dist[f"dir_far{int(100 * f)}"][name][start:start + len(idx)] = dir_at_far(models[name], counts, f)
            kvc = kvc_global(models[name], counts)
            dist["kvc_eer"][name][start:start + len(idx)], dist["kvc_fnmr1"][name][start:start + len(idx)] = kvc.T
    return dist


def point_estimates(models: dict[str, dict], n_people: int) -> dict:
    ones = np.ones((1, n_people))
    out = {}
    for name, s in models.items():
        out[name] = {m: float(v.mean()) for m, v in per_person(s).items()}
        for f in FAR_POINTS:
            out[name][f"dir_far{int(100 * f)}"] = float(dir_at_far(s, ones, f)[0])
        out[name]["kvc_eer"], out[name]["kvc_fnmr1"] = map(float, kvc_global(s, ones)[0])
    return out


def interval(x: np.ndarray) -> list[float]:
    return [round(float(np.percentile(x, 2.5)), 2), round(float(np.percentile(x, 97.5)), 2)]


def verdict(delta: np.ndarray) -> str:
    lo, hi = np.percentile(delta, [2.5, 97.5])
    if lo > 0:
        return "better"
    if hi < 0:
        return f"worse, but by less than {NON_INFERIORITY_MARGIN:g} pt" if lo > -NON_INFERIORITY_MARGIN else "worse"
    if lo > -NON_INFERIORITY_MARGIN:
        return f"not worse than {NON_INFERIORITY_MARGIN:g} pt, not better"
    return "inconclusive"


def summarize(models: dict[str, dict], n_people: int, B: int) -> dict:
    point, dist = point_estimates(models, n_people), bootstrap(models, n_people, B)
    reference = next(iter(models))
    report = {"people": n_people, "bootstrap": B, "reference": reference, "models": {}}
    for name in models:
        row = {m: [round(point[name][m], 2), *interval(dist[m][name])] for m in dist}
        if name != reference:
            row["vs_reference"] = {
                m: {"delta": round(point[name][m] - point[reference][m], 2),
                    "ci95": interval(dist[m][name] - dist[m][reference]),
                    "p_better": round(float(((dist[m][name] - dist[m][reference]) * (-1 if m.startswith(("eer", "kvc")) else 1) > 0).mean()), 3)}
                for m in dist}
            row["rank1_verdict"] = verdict(dist["rank1"][name] - dist["rank1"][reference])
        report["models"][name] = row
    # every model against every other, for the candidate-vs-candidate questions (e.g. floor cache or not)
    report["pairwise"] = {
        m: {a: {b: {"delta": round(point[a][m] - point[b][m], 2), "ci95": interval(dist[m][a] - dist[m][b])}
                for b in models if b != a} for a in models}
        for m in ("rank1", "eer_g1", "kvc_eer")}
    return report


def load_people_embeddings(model, data, device) -> tuple[torch.Tensor, list[str]]:
    """(N, 15, D) CPU tensor of each eligible person's first 15 sequence embeddings, people sorted by id."""
    by_person = session_embeddings(model, data, device)
    people = sorted(by_person)
    out = torch.empty(len(people), SEQUENCES, next(iter(by_person.values())).shape[1])
    for start in range(0, len(people), 5000):  # chunked so the stacked copy never sits on the GPU
        out[start:start + 5000] = torch.stack([by_person[p][:SEQUENCES] for p in people[start:start + 5000]]).cpu()
    del by_person
    torch.cuda.empty_cache()
    return out, people


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH", help="first one is the reference")
    parser.add_argument("--split", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--tag", required=True, help="result file is paired_<tag>.json")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--background", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_eval_data(args.split, args.cache)
    scores, people = {}, None
    for spec in args.model:
        name, path = spec.split("=", 1)
        embeddings, these = load_people_embeddings(load_any_encoder(path, device), data, device)
        assert people is None or these == people, "models must be scored on the same people"
        people = these
        scores[name] = score_model(embeddings, args.repetitions, args.background, device)
        print(f"scored {name} on {len(people)} people", flush=True)

    report = summarize(scores, len(people), args.bootstrap)
    report.update({"split": Path(args.split).as_posix(), "cache": Path(args.cache).as_posix(),
                   "repetitions": args.repetitions, "background": args.background,
                   "checkpoints": {s.split("=", 1)[0]: s.split("=", 1)[1] for s in args.model}})
    RESULTS_DIR.mkdir(exist_ok=True)
    json.dump(report, open(RESULTS_DIR / f"paired_{args.tag}.json", "w"), indent=2)

    print(f"\n{len(people)} people, {args.repetitions} repetitions, {args.bootstrap} bootstraps; [95% CI over people]")
    cols = ("rank1", "eer_g1", "eer_g10", "dir_far1", "dir_far5", "kvc_eer", "kvc_fnmr1")
    print(f"{'model':<18}" + "".join(f"{c:>26}" for c in cols) + "   Rank-1 vs reference")
    for name, row in report["models"].items():
        cells = "".join(f"{row[c][0]:>9.2f} [{row[c][1]:.2f},{row[c][2]:.2f}]".rjust(26) for c in cols)
        extra = ""
        if "vs_reference" in row:
            d = row["vs_reference"]["rank1"]
            extra = f"   {d['delta']:+.2f} [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}]  {row['rank1_verdict']}"
        print(f"{name:<18}{cells}{extra}")


if __name__ == "__main__":
    main()
