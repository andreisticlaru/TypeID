"""run with: python -m unittest typenet.test_losses -v"""

import unittest

import torch

from typenet.losses import mined_loss


class MinedLossTest(unittest.TestCase):
    # One anchor at the origin, its positive at squared distance 4. Pool = [positive, 3 negatives] with
    # squared distances 4 (same subject: must be skipped), 1, 9, 16.
    anchor = torch.tensor([[0.0, 0.0]])
    positive = torch.tensor([[2.0, 0.0]])
    negatives = torch.tensor([[1.0, 0.0], [3.0, 0.0], [4.0, 0.0]])

    def loss(self, mode, same_subject=(True, False, False, False), margin=1.5):
        return mined_loss(self.anchor, self.positive, self.negatives, torch.tensor([same_subject]), mode, margin)

    def test_hard_takes_closest_valid_negative(self):
        # closest valid: d=1 -> 4 - 1 + 1.5
        self.assertAlmostEqual(self.loss("hard").item(), 4.5)

    def test_semi_takes_closest_negative_farther_than_positive(self):
        # farther than d_ap=4: {9, 16} -> 9; margin 6 so 9 and 16 give different losses: 4 - 9 + 6 = 1
        self.assertAlmostEqual(self.loss("semi", margin=6.0).item(), 1.0)

    def test_semi_falls_back_to_farthest_valid(self):
        # only d=1 is valid and it is closer than the positive -> fallback picks it
        self.assertAlmostEqual(self.loss("semi", (True, False, True, True)).item(), 4.5)


if __name__ == "__main__":
    unittest.main()
