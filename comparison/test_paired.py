"""run with: python -m unittest comparison.test_paired -v"""

import unittest

import numpy as np
import torch

from comparison.paired import (batched_eer, bootstrap, dir_at_far, kvc_global, kvc_scores, partitions, person_eers,
                               score_model, summarize)
from eval.typenet_protocol import GALLERY_MAX, authentication_eer, identification_ranks, subject_eer

CPU = torch.device("cpu")


def people(n, noise, seed=0, dim=8):
    """(n, 15, dim): each person is a cluster centre plus noise, so a lower `noise` is a better embedding."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(n, 1, dim, generator=g) * 3
    return centres + noise * torch.randn(n, 15, dim, generator=g)


class PartitionTest(unittest.TestCase):
    def test_every_person_scored_once_with_full_backgrounds(self):
        for n in (60, 50, 20, 7):
            chunks = partitions(n, seed=3, size=20)
            scored = np.concatenate([c[:r] for c, r in chunks])
            self.assertEqual(sorted(scored), list(range(n)))
            for chunk, _ in chunks:
                self.assertEqual(len(chunk), min(20, n))
                self.assertEqual(len(set(chunk)), len(chunk))  # a top-up never repeats someone in the same background


class EerTest(unittest.TestCase):
    def test_batched_matches_subject_eer(self):
        g = torch.Generator().manual_seed(1)
        genuine, impostor = torch.rand(30, 5, generator=g), torch.rand(30, 40, generator=g) + 0.3
        expected = [subject_eer(genuine[i].numpy(), impostor[i].numpy()) for i in range(30)]
        np.testing.assert_allclose(batched_eer(genuine, impostor).numpy(), expected, atol=1e-6)

    def test_person_eers_mean_matches_authentication_eer(self):
        background = people(40, noise=2.0)
        for g in (1, 5, 10):
            ours = person_eers(background, g, np.random.default_rng(7)).mean()
            theirs = authentication_eer(background, g, np.random.default_rng(7))
            self.assertAlmostEqual(ours, theirs, places=4)


class ScoreTest(unittest.TestCase):
    def test_hit1_matches_identification_ranks(self):
        embeddings = people(30, noise=4.0)
        scores = score_model(embeddings, repetitions=1, size=30, device=CPU)
        chunk, _ = partitions(30, 0, 30)[0]
        ranks = identification_ranks(embeddings[torch.from_numpy(chunk)]).numpy()
        np.testing.assert_array_equal(scores["hit1"][0, chunk], ranks <= 1)

    def test_open_set_records_cover_enrolled_and_strangers(self):
        scores = score_model(people(40, noise=1.0), repetitions=2, size=20, device=CPU)
        self.assertEqual(len(scores["enrolled"]["person"]), 2 * 40 // 2)
        self.assertEqual(len(scores["strangers"]["person"]), 2 * 40 // 2)


class DirTest(unittest.TestCase):
    def test_threshold_is_set_by_strangers(self):
        scores = {"strangers": {"person": np.arange(4), "dist": np.array([1., 2., 3., 4.])},
                  "enrolled": {"person": np.arange(4), "correct": np.ones(4, bool), "dist": np.array([.5, 1.5, 2.5, 3.5])}}
        self.assertAlmostEqual(dir_at_far(scores, np.ones((1, 4)), 0.5)[0], 50.0)
        self.assertAlmostEqual(dir_at_far(scores, np.ones((1, 4)), 0.0)[0], 0.0)  # nothing accepted

    def test_wrong_name_never_counts(self):
        scores = {"strangers": {"person": np.arange(2), "dist": np.array([5., 6.])},
                  "enrolled": {"person": np.arange(2), "correct": np.array([True, False]), "dist": np.array([1., 1.])}}
        self.assertAlmostEqual(dir_at_far(scores, np.ones((1, 2)), 1.0)[0], 50.0)


class KvcTest(unittest.TestCase):
    def test_scores_shapes_and_genuine_closer(self):
        genuine, impostor = kvc_scores(people(30, noise=1.0), np.random.default_rng(0))
        self.assertEqual(genuine.shape, (30, 10))
        self.assertEqual(impostor.shape, (30, 20))
        self.assertLess(genuine.mean(), impostor.mean())

    def test_never_its_own_impostor(self):
        background = people(25, noise=0.0)  # zero noise: only a self-comparison could give distance 0
        _, impostor = kvc_scores(background, np.random.default_rng(1))
        self.assertGreater(impostor.min(), 0)

    def test_unit_weights_match_pooled_eer(self):
        rng = np.random.default_rng(2)
        genuine, impostor = rng.random(40), rng.random(80) + 0.3
        scores = {"kvc": {"genuine": genuine, "genuine_person": np.arange(40) % 4,
                          "impostor": impostor, "impostor_person": np.arange(80) % 4}}
        self.assertAlmostEqual(kvc_global(scores, np.ones((1, 4)))[0, 0], 100 * subject_eer(genuine, impostor), places=4)

    def test_weight_two_equals_duplicated_person(self):
        rng = np.random.default_rng(3)
        g, i = rng.random(6), rng.random(6) + 0.2
        gp, ip = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 0, 1, 1, 2, 2])
        scores = {"kvc": {"genuine": g, "genuine_person": gp, "impostor": i, "impostor_person": ip}}
        doubled = {"kvc": {"genuine": np.r_[g, g[:2]], "genuine_person": np.r_[gp, [0, 0]],
                           "impostor": np.r_[i, i[:2]], "impostor_person": np.r_[ip, [0, 0]]}}
        np.testing.assert_allclose(kvc_global(scores, np.array([[2, 1, 1]])), kvc_global(doubled, np.ones((1, 3))))


class VerdictWordsTest(unittest.TestCase):
    def test_small_but_certain_loss_is_not_called_plain_worse(self):
        from comparison.paired import verdict
        rng = np.random.default_rng(0)
        self.assertEqual(verdict(rng.normal(-0.5, 0.1, 500)), "worse, but by less than 1 pt")
        self.assertEqual(verdict(rng.normal(-3.0, 0.1, 500)), "worse")
        self.assertEqual(verdict(rng.normal(0.5, 0.1, 500)), "better")
        self.assertEqual(verdict(rng.normal(-0.2, 0.2, 500)), "not worse than 1 pt, not better")
        self.assertEqual(verdict(rng.normal(-0.2, 0.5, 500)), "inconclusive")


class VerdictTest(unittest.TestCase):
    def setUp(self):
        self.n = 60
        self.models = {name: score_model(people(self.n, noise), repetitions=2, size=20, device=CPU)
                       for name, noise in (("ref", 4.0), ("same", 4.0), ("better", 1.0))}

    def test_identical_model_is_a_zero_difference(self):
        report = summarize(self.models, self.n, B=200)
        delta = report["models"]["same"]["vs_reference"]["rank1"]
        self.assertEqual(delta["delta"], 0.0)
        self.assertEqual(delta["ci95"], [0.0, 0.0])

    def test_pairwise_is_antisymmetric(self):
        pair = summarize(self.models, self.n, B=200)["pairwise"]["rank1"]
        self.assertEqual(pair["better"]["ref"]["delta"], -pair["ref"]["better"]["delta"])

    def test_cleaner_embedding_is_better(self):
        report = summarize(self.models, self.n, B=200)["models"]["better"]
        self.assertEqual(report["rank1_verdict"], "better")
        self.assertLess(report["vs_reference"]["eer_g1"]["delta"], 0)  # lower EER is the improvement
        self.assertLess(report["vs_reference"]["kvc_eer"]["delta"], 0)
        self.assertGreater(report["vs_reference"]["kvc_eer"]["p_better"], 0.5)


if __name__ == "__main__":
    unittest.main()
