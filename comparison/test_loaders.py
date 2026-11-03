import unittest

import torch

from comparison.loaders import PauseEdit


class Spy(torch.nn.Module):
    def forward(self, x, mask):
        return x


def window():
    # one pause of 5,000 ms IL at step 1; step 3 is padding
    x = torch.tensor([[[100., 50., 150., 160.], [100., 5000., 5100., 5100.], [90., 40., 130., 140.], [0., 0., 0., 0.]]])
    return x, torch.tensor([[True, True, True, False]])


class PauseEditTest(unittest.TestCase):
    def test_cap_shortens_one_pause_in_il_pl_rl_together(self):
        x, mask = window()
        out = PauseEdit(Spy(), "cap", cap=1000.)(x, mask)
        self.assertEqual(out[0, 1].tolist(), [100., 1000., 1100., 1100.])
        self.assertTrue(torch.equal(out[0, [0, 2, 3]], x[0, [0, 2, 3]]))

    def test_soft_cap_keeps_order_above_the_cap(self):
        x, mask = window()
        out = PauseEdit(Spy(), "soft", cap=1000.)(x, mask)
        self.assertGreater(out[0, 1, 1], 1000.)
        self.assertLess(out[0, 1, 1], 5000.)
        longer = x.clone()
        longer[0, 1, 1:] += 50000.
        self.assertGreater(PauseEdit(Spy(), "soft", cap=1000.)(longer, mask)[0, 1, 1], out[0, 1, 1])

    def test_noise_moves_the_same_amount_and_never_touches_padding_or_holds(self):
        x, mask = window()
        for _ in range(20):
            out = PauseEdit(Spy(), "noise", cap=1000.)(x, mask)
            self.assertTrue(torch.equal(out[..., 0], x[..., 0]))
            self.assertTrue(torch.equal(out[0, 3], x[0, 3]))
            self.assertAlmostEqual((out - x)[..., 1].abs().sum().item(), 4000.)


if __name__ == "__main__":
    unittest.main()
