"""run with: python -m unittest typenet.test_network -v"""

import unittest

import torch

from typenet.network import TypeNetEncoder


def batch(n=8, length=30, m=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, m, 4, generator=g) * 300
    mask = torch.zeros(n, m, dtype=torch.bool)
    mask[:, :length] = True
    x[~mask] = 0
    return x, mask


class TypeNetEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = TypeNetEncoder()

    def test_padding_cannot_leak_in_eval(self):
        self.model.eval()
        x, mask = batch()
        noisy = x.clone()
        noisy[~mask] = torch.rand(int((~mask).sum()), 4) * 1e4
        torch.testing.assert_close(self.model(x, mask), self.model(noisy, mask))

    def test_padding_cannot_leak_into_batchnorm_statistics(self):
        model = TypeNetEncoder(dropout=0.0, recurrent_dropout=0.0).train()
        x, mask = batch()
        noisy = x.clone()
        noisy[~mask] = torch.rand(int((~mask).sum()), 4) * 1e4
        torch.testing.assert_close(model(x, mask), model(noisy, mask))

    def test_grouped_pass_equals_separate_calls(self):
        """Training runs anchor/positive/negative as one batch with groups=3; it must match three calls,
        including the BatchNorm running averages it leaves behind."""
        separate = TypeNetEncoder(dropout=0.0, recurrent_dropout=0.0).train()
        grouped = TypeNetEncoder(dropout=0.0, recurrent_dropout=0.0).train()
        grouped.load_state_dict(separate.state_dict())
        parts = [batch(length=20 + 10 * k, seed=k) for k in range(3)]
        expected = torch.cat([separate(x, m) for x, m in parts])
        actual = grouped(torch.cat([x for x, _ in parts]), torch.cat([m for _, m in parts]), groups=3)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(grouped.buffers(), separate.buffers()):
            torch.testing.assert_close(a, b)

    def test_readout_is_last_real_step(self):
        self.model.eval()
        x, mask = batch(length=30)
        unpadded = self.model(x[:, :30], mask[:, :30])
        torch.testing.assert_close(self.model(x, mask), unpadded)

    def test_parameter_count(self):
        """Trainable: BN(4) 8 + LSTM(4->128) 4*128*(4+128+1) = 68,096 + BN(128) 256
        + LSTM(128->128) 4*128*(128+128+1) = 131,584 -> 199,944. The paper reports 200,458 trainable
        parameters: the same sum with its 5th input (key code), i.e. + 2 (BN) + 512 (one more LSTM input
        column). Buffers are BatchNorm's running mean/variance (8 + 256)."""
        trainable = sum(p.numel() for p in self.model.parameters())
        buffers = sum(b.numel() for b in self.model.buffers())
        self.assertEqual(trainable, 199_944)
        self.assertEqual(trainable + buffers, 200_208)


if __name__ == "__main__":
    unittest.main()
