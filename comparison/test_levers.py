import unittest

import numpy as np
import torch

from comparison.levers import pool, sequence_distances, snorm


class PoolTest(unittest.TestCase):
    def test_k1_is_identity(self):
        g, i = np.arange(20.).reshape(2, 10), np.arange(60.).reshape(2, 3, 10)
        pick = np.array([[0, 4, 9], [1, 2, 3]])
        pg, pi = pool(g, i, 1, pick)
        np.testing.assert_array_equal(pg, g)
        np.testing.assert_array_equal(pi, np.take_along_axis(i, pick[:, :, None], axis=2)[:, :, 0])

    def test_k3_means_consecutive_groups(self):
        g = np.arange(10.)[None]
        i = np.arange(10.)[None, None]
        pg, pi = pool(g, i, 3, np.array([[4]]))  # 3 groups, pick 4 % 3 = group 1 = sequences 3, 4, 5
        np.testing.assert_array_equal(pg, [[1., 4., 7.]])
        np.testing.assert_array_equal(pi, [[4.]])


class SnormTest(unittest.TestCase):
    def test_standardizes_from_both_sides(self):
        g, i = np.array([[3.]]), np.array([[[5.]]])
        stats = (np.array([1.]), np.array([2.]), np.array([[3.]]), np.array([[1.]]))  # mean_e, std_e, mean_q, std_q
        sg, si = snorm(g, i, np.array([[0]]), stats)
        self.assertAlmostEqual(sg[0, 0], ((3 - 1) / 2 + (3 - 3) / 1) / 2)
        self.assertAlmostEqual(si[0, 0, 0], ((5 - 1) / 2 + (5 - 3) / 1) / 2)


class SequenceDistancesTest(unittest.TestCase):
    def test_matches_direct_computation(self):
        emb = torch.randn(4, 15, 8)
        scored, others = np.array([0, 2]), np.array([[1, 3], [3, 0]])
        g, i = sequence_distances(emb, scored, others, torch.device("cpu"))
        want_g = torch.cdist(emb[2, :5], emb[2, 5:]).mean(dim=0)
        want_i = torch.cdist(emb[2, :5], emb[0, 5:]).mean(dim=0)
        np.testing.assert_allclose(g[1], want_g.numpy(), rtol=1e-5)
        np.testing.assert_allclose(i[1, 1], want_i.numpy(), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
