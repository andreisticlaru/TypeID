import unittest

from data_v2.keyrecs import CHUNK
from data_v2.klicke import BURST, LAYOUTS, first_chunks, layouts_for, log_to_pairs, pause_free_pieces, split_people

HEADER = '"","DownEventID","UpEventID","DownTime","UpTime","ActionTime","DownEvent","UpEvent","CursorPosition",' \
         '"PauseTime","WordCount","TextChange","Activity"\n'


def row(down, up, event):
    return f'"1",1,1,{down},{up},{up - down},"{event}","{event}",0,0,0,"x","Input"\n'


def keys(n, start=0.0):
    return [(start + 10.0 * i, start + 10.0 * i + 5) for i in range(n)]


class LogToPairsTest(unittest.TestCase):
    def test_keeps_real_keys(self):
        self.assertEqual(log_to_pairs(HEADER + row(100, 180, "a") + row(150, 260, "Space")), [(100, 180), (150, 260)])

    def test_drops_mouse_but_the_gap_stays(self):
        pairs = log_to_pairs(HEADER + row(0, 50, "a") + row(100, 200, "Leftclick") + row(900, 950, "b"))
        self.assertEqual(pairs, [(0, 50), (900, 950)])

    def test_drops_untimed_modifiers(self):
        # Shift's up time equals its down time in every row of the corpus
        pairs = log_to_pairs(HEADER + row(0, 50, "a") + row(70, 70, "Shift") + row(90, 150, "B"))
        self.assertEqual(pairs, [(0, 50), (90, 150)])

    def test_skips_broken_rows(self):
        self.assertEqual(log_to_pairs(HEADER + '"1",bad\n' + row(0, 50, "a")), [(0, 50)])


class ChunkingTest(unittest.TestCase):
    def test_chunks_never_span_two_tasks(self):
        self.assertEqual(first_chunks([keys(CHUNK + 50), keys(CHUNK + 1, 1e6)], 3), [])
        got = first_chunks([keys(CHUNK + 50), keys(2 * CHUNK, 1e6)], 3)
        self.assertEqual([c[0][0] for c in got], [0.0, 1e6, 1e6 + 10.0 * CHUNK])


class LayoutsTest(unittest.TestCase):
    def setUp(self):
        self.copy = [keys(3 * CHUNK, 1e6 * t) for t in range(1, 6)]
        self.essay = keys(40 * CHUNK, 1e8)
        self.built = layouts_for(self.copy, self.essay)

    def test_every_layout_has_15_full_chunks(self):
        self.assertEqual(set(self.built), {*LAYOUTS, BURST})
        for sequences in self.built.values():
            self.assertEqual(len(sequences), 15)
            self.assertTrue(all(len(c) == CHUNK for c in sequences))

    def test_each_comparison_changes_one_thing(self):
        b = self.built
        self.assertEqual(b["copy_essay"][:5], b["copy"][:5])  # same enrollment
        self.assertEqual(b["copy_essay_id"][:10], b["copy"][:10])  # same gallery
        self.assertEqual(b["essay"][5:], b["copy_essay"][5:])  # same queries
        self.assertEqual(b["copy_essay_id"][10:], b["copy_essay"][10:])

    def test_essay_enrollment_comes_before_its_queries(self):
        essay = self.built["essay"]
        self.assertLess(essay[4][-1][0], essay[5][0][0])

    def test_short_copy_only_loses_copy_layouts(self):
        built = layouts_for([keys(5 * CHUNK)], self.essay)
        self.assertEqual(set(built), {"copy_essay", "essay", BURST})

    def test_burst_shares_enrollment_with_copy(self):
        self.assertEqual(self.built[BURST][:5], self.built["copy"][:5])


class PauseFreeTest(unittest.TestCase):
    def test_skips_past_a_long_pause(self):
        first = keys(10)
        rest = keys(CHUNK, first[-1][1] + 5000)  # a 5 s gap after key 10
        got = pause_free_pieces(first + rest)
        self.assertEqual([p[0] for p in got], [rest[0]])

    def test_short_gaps_are_kept_and_whole(self):
        got = pause_free_pieces(keys(2 * CHUNK + 3))
        self.assertEqual(len(got), 2)
        self.assertTrue(all(len(p) == CHUNK for p in got))


class SplitTest(unittest.TestCase):
    def test_disjoint_stable_and_complete(self):
        people = [str(i) for i in range(101)]
        dev, test = split_people(people)
        self.assertFalse(dev & test)
        self.assertEqual(dev | test, set(people))
        self.assertEqual(split_people(list(reversed(people))), (dev, test))


if __name__ == "__main__":
    unittest.main()
