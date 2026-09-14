"""Unit tests for the canonical feature extractor.

Run with: python -m unittest features.test_extract -v
"""

import unittest

from features.extract import (
    M,
    MIN_KEYSTROKES,
    extract_features,
    pair_browser_events,
    windows_from_keystrokes,
)


def make_pairs(n, dwell=100.0, gap=50.0, start=0.0):
    """n synthetic keystrokes with uniform HL=dwell, IL=PL-dwell=gap, PL=RL=dwell+gap."""
    pairs = []
    t = start
    for _ in range(n):
        press = t
        release = press + dwell
        pairs.append((press, release))
        t = release + gap
    return pairs


class WindowsFromKeystrokesTests(unittest.TestCase):
    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            windows_from_keystrokes([])

    def test_rejects_below_minimum(self):
        pairs = make_pairs(MIN_KEYSTROKES)  # one short of the +1 required
        with self.assertRaises(ValueError):
            windows_from_keystrokes(pairs)

    def test_accepts_exact_minimum(self):
        pairs = make_pairs(MIN_KEYSTROKES + 1)  # boundary: must NOT raise
        windows, mask = windows_from_keystrokes(pairs)
        self.assertEqual(windows.shape, (1, M, 4))
        self.assertEqual(mask.shape, (1, M))
        self.assertEqual(mask.sum(), MIN_KEYSTROKES)  # N-1 real vectors from N keystrokes

    def test_feature_arithmetic(self):
        # dwell=100, gap=50 -> HL=100, IL=50, PL=RL=150 for every keystroke
        pairs = make_pairs(MIN_KEYSTROKES + 1, dwell=100.0, gap=50.0)
        windows, mask = windows_from_keystrokes(pairs)
        real = windows[0][mask[0]]
        for hl, il, pl, rl in real:
            self.assertAlmostEqual(hl, 100.0)
            self.assertAlmostEqual(il, 50.0)
            self.assertAlmostEqual(pl, 150.0)
            self.assertAlmostEqual(rl, 150.0)

    def test_negative_inter_key_latency_is_kept(self):
        # Key rollover: next key pressed before current one releases (IL < 0).
        # The Python extractor must NOT filter this out -- unlike the frontend's
        # display-only JS computation, this is real signal for the model.
        pairs = make_pairs(MIN_KEYSTROKES + 1)  # default gap=50 between press[6] and release[5]
        press, release = pairs[5]
        pairs[5] = (press, release + 70.0)  # push release past press[6] to force IL(5) < 0

        windows, mask = windows_from_keystrokes(pairs)
        il_values = windows[0][:, 1][mask[0]]
        self.assertTrue((il_values < 0).any())

    def test_exact_single_window_no_padding(self):
        # N-1 == M exactly: one full window, nothing padded.
        pairs = make_pairs(M + 1)
        windows, mask = windows_from_keystrokes(pairs)
        self.assertEqual(windows.shape, (1, M, 4))
        self.assertTrue(mask.all())

    def test_short_trailing_window_is_dropped(self):
        # N-1 == M + 1: second window would hold only one real vector,
        # well under MIN_KEYSTROKES -- too thin to trust, so it's dropped
        # rather than kept as a near-empty window.
        pairs = make_pairs(M + 2)
        windows, mask = windows_from_keystrokes(pairs)
        self.assertEqual(windows.shape, (1, M, 4))
        self.assertTrue(mask.all())

    def test_splits_into_multiple_windows(self):
        # N-1 == M + MIN_KEYSTROKES: second window holds exactly
        # MIN_KEYSTROKES real vectors, right at the floor, so it's kept.
        pairs = make_pairs(M + MIN_KEYSTROKES + 1)
        windows, mask = windows_from_keystrokes(pairs)
        self.assertEqual(windows.shape, (2, M, 4))
        self.assertTrue(mask[0].all())
        self.assertEqual(mask[1].sum(), MIN_KEYSTROKES)
        self.assertTrue(mask[1][:MIN_KEYSTROKES].all())
        self.assertFalse(mask[1][MIN_KEYSTROKES:].any())
        # padded slots must be zero, not garbage
        self.assertTrue((windows[1][~mask[1]] == 0).all())


class PairBrowserEventsTests(unittest.TestCase):
    def test_simple_sequential_pairs(self):
        events = [
            {"code": "KeyA", "key": "a", "type": "keydown", "t": 0},
            {"code": "KeyA", "key": "a", "type": "keyup", "t": 100},
            {"code": "KeyB", "key": "b", "type": "keydown", "t": 150},
            {"code": "KeyB", "key": "b", "type": "keyup", "t": 260},
        ]
        self.assertEqual(pair_browser_events(events), [(0, 100), (150, 260)])

    def test_overlapping_keys_pair_by_code_not_arrival_order(self):
        # Shift held down while W is pressed and released underneath it.
        events = [
            {"code": "ShiftLeft", "key": "Shift", "type": "keydown", "t": 0},
            {"code": "KeyW", "key": "w", "type": "keydown", "t": 50},
            {"code": "KeyW", "key": "w", "type": "keyup", "t": 150},
            {"code": "ShiftLeft", "key": "Shift", "type": "keyup", "t": 200},
        ]
        # sorted by press time: Shift (0,200) before W (50,150)
        self.assertEqual(pair_browser_events(events), [(0, 200), (50, 150)])

    def test_orphaned_keyup_is_dropped_not_crashed(self):
        events = [
            {"code": "KeyA", "key": "a", "type": "keyup", "t": 100},  # no matching keydown
            {"code": "KeyB", "key": "b", "type": "keydown", "t": 150},
            {"code": "KeyB", "key": "b", "type": "keyup", "t": 260},
        ]
        self.assertEqual(pair_browser_events(events), [(150, 260)])


class ExtractFeaturesIntegrationTests(unittest.TestCase):
    def test_matches_manual_pairing_plus_windowing(self):
        events = []
        t = 0.0
        for i in range(MIN_KEYSTROKES + 1):
            code = f"Key{i}"
            events.append({"code": code, "key": code, "type": "keydown", "t": t})
            events.append({"code": code, "key": code, "type": "keyup", "t": t + 100})
            t += 150

        got_windows, got_mask = extract_features(events)
        want_windows, want_mask = windows_from_keystrokes(pair_browser_events(events))

        self.assertTrue((got_windows == want_windows).all())
        self.assertTrue((got_mask == want_mask).all())


if __name__ == "__main__":
    unittest.main()
