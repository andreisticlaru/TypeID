import unittest

import numpy as np

from comparison.decisions import SPRT_BOUND, cllr, fit_calibration, sprt


class CalibrationTest(unittest.TestCase):
    def test_recovers_the_true_llr_of_two_gaussians(self):
        rng = np.random.default_rng(0)
        genuine, impostor = rng.normal(0, 1, 200000), rng.normal(2, 1, 200000)
        a, b = fit_calibration(genuine, impostor)  # true LLR = -2x + 2
        self.assertAlmostEqual(a, -2, delta=0.05)
        self.assertAlmostEqual(b, 2, delta=0.05)

    def test_balanced_despite_unequal_class_sizes(self):
        rng = np.random.default_rng(1)
        a, b = fit_calibration(rng.normal(0, 1, 5000), rng.normal(2, 1, 100000))
        self.assertAlmostEqual(-b / a, 1, delta=0.1)  # LLR = 0 half way between the classes


class CllrTest(unittest.TestCase):
    def test_uninformative_costs_one_bit(self):
        self.assertAlmostEqual(cllr(np.zeros(5), np.zeros(5)), 1.0)

    def test_correct_is_near_zero_and_misleading_is_above_one(self):
        self.assertLess(cllr(np.full(5, 20.), np.full(5, -20.)), 1e-6)
        self.assertGreater(cllr(np.full(5, -5.), np.full(5, 5.)), 1)


class SprtTest(unittest.TestCase):
    def test_lock_out_accept_and_undecided(self):
        step = SPRT_BOUND / 2 + 0.01
        llrs = np.array([[-step, -step, 0, 0], [step, step, -9 * step, 0], [0, 0, 0, 0.]])
        until, accepted = sprt(llrs)
        self.assertEqual(until.tolist(), [2, 4, 4])  # accepted and undecided both count as never locked out
        self.assertEqual(accepted.tolist(), [False, True, False])


if __name__ == "__main__":
    unittest.main()
