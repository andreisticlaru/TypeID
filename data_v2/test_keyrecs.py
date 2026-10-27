import unittest

from data_v2.keyrecs import chunks, parse, runs_to_pairs


class RunsToPairsTest(unittest.TestCase):
    def test_rebuilds_absolute_times(self):
        # a: press 0, release 100; b: press 150, release 230; c: press 300, release 420 (ms)
        rows = [["p", "1", "a", "b", "0.1", "0.15", "0.23"], ["p", "1", "b", "c", "0.08", "0.15", "0.27"]]
        (pairs,) = runs_to_pairs(map(parse, rows))
        expected = [(0, 100), (150, 230), (300, 420)]
        for got, want in zip(pairs, expected, strict=True):
            self.assertAlmostEqual(got[0], want[0])
            self.assertAlmostEqual(got[1], want[1])

    def test_rollover_is_sorted_by_press(self):
        # Shift pressed 0.8 s before W (negative DD), as in KeyRecs' first row
        (pairs,) = runs_to_pairs([parse(["p", "1", "W", "Shift", "0.15", "-0.8", "0.16"])])
        self.assertLess(pairs[0][0], pairs[1][0])
        self.assertAlmostEqual(pairs[0][0], -800)

    def test_break_splits_runs(self):
        rows = [["p", "1", "a", "b", "0.1", "0.2", "0.3"], ["p", "1", "x", "y", "0.1", "0.2", "0.3"],
                ["p", "1", "y", "z", "", "", ""]]
        runs = runs_to_pairs(map(parse, rows))
        self.assertEqual([len(r) for r in runs], [2, 2])

    def test_chunks_evenly_spaced_and_whole(self):
        run = [(float(i), i + 0.5) for i in range(51 * 10)]
        got = chunks([run], 4)
        self.assertEqual([c[0][0] for c in got], [0.0, 153.0, 306.0, 459.0])
        self.assertTrue(all(len(c) == 51 for c in got))
        self.assertEqual(chunks([run], 11), [])


if __name__ == "__main__":
    unittest.main()
