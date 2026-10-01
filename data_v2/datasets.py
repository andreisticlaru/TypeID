"""Training sampler and eval data for any (split, cache) pair.

model.train.build_sampler and eval.rank_n.load_eval_data hard-code v1's split and cache; these take both as
arguments and return the same objects, so the sampler and every eval function work unchanged on either.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model.triplet_sampler import TripletSampler

ARRAYS = ("windows", "mask", "subject_ids", "session_ids")


def load_split(split_path) -> dict:
    with open(split_path) as f:
        return json.load(f)


def make_sampler(split_path, cache_dir) -> TripletSampler:
    """TripletSampler over the split's train subjects only; eval subjects are never indexed."""
    cache_dir = Path(cache_dir)
    return TripletSampler(*(cache_dir / f"{name}.npy" for name in ARRAYS),
                          subjects=set(load_split(split_path)["train_subjects"]))


def load_eval_data(split_path, cache_dir):
    """(windows, mask, eval_rows, rows_by_session) for the split's eval subjects; same as eval.rank_n.load_eval_data."""
    windows, mask, subject_ids, session_ids = (np.load(Path(cache_dir) / f"{name}.npy") for name in ARRAYS)
    eval_rows = np.flatnonzero(np.isin(subject_ids, load_split(split_path)["eval_subjects"]))  # ascending
    rows_by_session: dict[tuple, list[int]] = {}
    for i in eval_rows:
        rows_by_session.setdefault((subject_ids[i], session_ids[i]), []).append(i)
    return windows, mask, eval_rows, rows_by_session
