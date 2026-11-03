import unittest

import torch
import torch.nn.functional as F

from comparison.backend import center, condition_means, score_variants, variants, wccn_map


def people(n=40, seed=0, shift=0.0):
    """Unit-ish embeddings: a per-person identity in the first dims, noise in dim 7, queries shifted along dim 0."""
    g = torch.Generator().manual_seed(seed)
    identity = torch.randn(n, 1, 8, generator=g)
    identity[..., 7] = 0
    noise = torch.randn(n, 15, 8, generator=g) * 0.05
    noise[..., 7] *= 40
    emb = identity + noise
    emb[:, 5:, 0] += shift
    return emb


class FitTest(unittest.TestCase):
    def test_center_removes_a_shared_shift(self):
        fit, emb = people(shift=3.0), people(seed=1, shift=3.0)
        centered = center(emb, condition_means(fit))
        gap = centered[:, 5:, 0].mean() - centered[:, :5, 0].mean()
        self.assertLess(gap.abs().item(), 0.5)

    def test_wccn_shrinks_the_noisy_direction(self):
        fit = people()
        w = wccn_map(fit)
        noisy, quiet = torch.zeros(8), torch.zeros(8)
        noisy[7], quiet[1] = 1, 1
        self.assertLess((noisy @ w).norm(), (quiet @ w).norm())

    def test_variants_keep_shape_and_unit_length(self):
        fit, emb = people(), people(seed=1)
        out = variants(fit, emb)
        self.assertEqual(set(out), {"cosine", "center", "wccn", "center+wccn"})
        for name in ("center", "wccn", "center+wccn"):
            self.assertEqual(out[name].shape, emb.shape)
            self.assertTrue(torch.allclose(out[name].norm(dim=-1), torch.ones(40, 15), atol=1e-4))


class ScoreTest(unittest.TestCase):
    def test_center_beats_cosine_under_a_shared_shift(self):
        fit, emb = people(n=60, shift=3.0), people(n=60, seed=1, shift=3.0)
        by = variants(F.normalize(fit, dim=-1), F.normalize(emb, dim=-1))
        report = score_variants({k: by[k] for k in ("cosine", "center")}, 2, 10, 20, torch.device("cpu"))
        self.assertEqual(report["variants"]["cosine"]["delta_vs_cosine"], 0)
        self.assertIn("center snorm", report["variants"])
        self.assertLess(report["variants"]["center"]["kvc_eer"], report["variants"]["cosine"]["kvc_eer"])


if __name__ == "__main__":
    unittest.main()
