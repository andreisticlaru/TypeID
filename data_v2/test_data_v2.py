"""Checks on the TypeNet split and the no-floor cache (needs data/split_typenet.json and data/preprocessed_typenet/).

run with: python -m unittest data_v2.test_data_v2 -v
"""

import unittest

import numpy as np

from data_v2.build_cache import CACHE_PATH
from data_v2.datasets import load_split, make_sampler
from data_v2.make_split import SPLIT_PATH, TRAIN_SUBJECTS
from features.extract import M


class TypeNetSplitTest(unittest.TestCase):
    def test_split_is_disjoint_and_numeric_prefix(self):
        split = load_split(SPLIT_PATH)
        train, held_out = split["train_subjects"], split["eval_subjects"]
        self.assertEqual(len(train), TRAIN_SUBJECTS)
        self.assertFalse(set(train) & set(held_out))
        self.assertLess(max(map(int, train)), min(map(int, held_out)))


class TypeNetCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subject_ids = np.load(CACHE_PATH / "subject_ids.npy")
        cls.session_ids = np.load(CACHE_PATH / "session_ids.npy")
        cls.mask = np.load(CACHE_PATH / "mask.npy", mmap_mode="r")

    def test_one_row_per_session(self):
        keys = np.char.add(np.char.add(self.subject_ids, "_"), self.session_ids)
        self.assertEqual(len(np.unique(keys)), len(keys))

    def test_no_floor_and_left_aligned(self):
        lengths = np.asarray(self.mask[:200_000]).sum(1)
        self.assertGreaterEqual(lengths.min(), 1)
        self.assertLessEqual(lengths.max(), M)
        self.assertLess(lengths.min(), 25)  # the v1 floor is not applied
        self.assertTrue((np.asarray(self.mask[:200_000]).argmin(1)[lengths < M] == lengths[lengths < M]).all())


class SamplerTest(unittest.TestCase):
    def test_sampler_never_yields_eval_subjects(self):
        held_out = set(load_split(SPLIT_PATH)["eval_subjects"])
        sampler = make_sampler(SPLIT_PATH, CACHE_PATH)
        self.assertFalse(set(sampler.all_subject_ids) & held_out)
        np.random.seed(0)
        for _ in range(50):
            batch = sampler.sample_batch(64)
            self.assertFalse(set(batch["anchor_subjects"]) & held_out)
            self.assertFalse(set(batch["negative_subjects"]) & held_out)


if __name__ == "__main__":
    unittest.main()
