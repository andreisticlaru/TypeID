"""run with: python -m unittest typenet.test_network -v"""

import itertools
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



FLAGS = {"normalize": True, "readout": "mean", "batchnorm": False}  # each one's non-default value


class AblationFlagsTest(unittest.TestCase):
    def build(self, **flags):
        torch.manual_seed(0)
        return TypeNetEncoder(dropout=0.0, recurrent_dropout=0.0, **flags)

    def test_each_flag_changes_the_output(self):
        x, mask = batch(length=30)
        baseline = self.build().eval()(x, mask)
        for name, value in FLAGS.items():
            with self.subTest(flag=name):
                self.assertFalse(torch.allclose(self.build(**{name: value}).eval()(x, mask), baseline))

    def test_padding_invariance_for_every_combination(self):
        x, mask = batch(length=30)
        mask[:3, 12:] = False  # mixed lengths, so the mean readout and BN statistics are exercised
        x[~mask] = 0
        noisy = x.clone()
        noisy[~mask] = torch.rand(int((~mask).sum()), 4) * 1e4
        for values in itertools.product([False, True], repeat=len(FLAGS)):
            flags = {name: FLAGS[name] for name, on in zip(FLAGS, values) if on}
            for mode in ("train", "eval"):
                with self.subTest(flags=flags, mode=mode):
                    model = getattr(self.build(**flags), mode)()
                    torch.testing.assert_close(model(x, mask), model(noisy, mask))

    def test_normalize_gives_unit_vectors(self):
        x, mask = batch()
        out = self.build(normalize=True).eval()(x, mask)
        torch.testing.assert_close(out.norm(dim=1), torch.ones(len(out)))

    def test_mean_readout_averages_real_steps(self):
        model = self.build(readout="mean").eval()
        x, mask = batch(n=2, length=10)
        steps =model.lstm2(model.bn2(model.lstm1(model.bn1(x / model.input_scale, mask)), mask))
        torch.testing.assert_close(model(x, mask), steps[:, :10].mean(1))

    def test_no_batchnorm_removes_its_parameters(self):
        model = self.build(batchnorm=False)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 199_944 - 8 - 256)
        self.assertEqual(list(model.buffers()), [])


if __name__ == "__main__":
    unittest.main()
