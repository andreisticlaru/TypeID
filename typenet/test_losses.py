"""run with: python -m unittest typenet.test_losses -v"""

import unittest

import torch

from typenet.losses import mined_loss, triplet_loss


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



class CosineMinedLossTest(unittest.TestCase):
    # Unit vectors. Anchor (1, 0); positive at cosine distance 0.4. Pool = [positive (same subject), negatives at
    # cosine distances 0.2, 1.6, 2.0].
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[0.6, 0.8]])
    negatives = torch.tensor([[0.8, 0.6], [-0.6, 0.8], [-1.0, 0.0]])

    def loss(self, mode, same_subject=(True, False, False, False), margin=0.5):
        return mined_loss(self.anchor, self.positive, self.negatives, torch.tensor([same_subject]), mode, margin,
                          distance="cosine")

    def test_hard_takes_closest_valid_negative(self):
        self.assertAlmostEqual(self.loss("hard").item(), 0.4 - 0.2 + 0.5, places=5)

    def test_semi_takes_closest_negative_farther_than_positive(self):
        # farther than 0.4: {1.6, 2.0} -> 1.6; margin 2 so the two picks give different losses
        self.assertAlmostEqual(self.loss("semi", margin=2.0).item(), 0.4 - 1.6 + 2.0, places=5)

    def test_semi_falls_back_to_farthest_valid(self):
        self.assertAlmostEqual(self.loss("semi", (True, False, True, True)).item(), 0.4 - 0.2 + 0.5, places=5)

    def test_random_triplet_loss_matches_v1_rule(self):
        from model.losses import triplet_loss as v1_triplet_loss
        negative = self.negatives[:1]
        self.assertAlmostEqual(triplet_loss(self.anchor, self.positive, negative, 0.5, distance="cosine").item(),
                               v1_triplet_loss(self.anchor, self.positive, negative, 0.5).item(), places=6)


if __name__ == "__main__":
    unittest.main()
