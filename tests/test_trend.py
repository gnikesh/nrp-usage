"""Unit tests for rolling token-production history."""

from __future__ import annotations

import unittest

from nrp_usage.trend import FLAT_BAND, Interval, Trend, TrendTracker, classify, combine

IN, OUT = "input", "output"
ROW = ("acme", "main", "qwen3")
K_IN, K_OUT = (ROW[0], ROW[1], ROW[2], IN), (ROW[0], ROW[1], ROW[2], OUT)


def iv(tokens=0.0, per=0.0, seconds=5.0, trend=Trend.UNKNOWN):
    return Interval(tokens, per, seconds, trend)


class TestClassify(unittest.TestCase):
    def test_no_predecessor_is_unknown(self):
        self.assertIs(classify(10.0, None), Trend.UNKNOWN)

    def test_up_down_flat(self):
        self.assertIs(classify(20.0, 10.0), Trend.UP)
        self.assertIs(classify(5.0, 10.0), Trend.DOWN)
        self.assertIs(classify(10.2, 10.0), Trend.FLAT)

    def test_dead_band_edges(self):
        self.assertIs(classify(10.0 * (1 + FLAT_BAND), 10.0), Trend.FLAT)
        self.assertIs(classify(10.0 * (1 + FLAT_BAND) + 0.01, 10.0), Trend.UP)
        self.assertIs(classify(10.0 * (1 - FLAT_BAND) - 0.01, 10.0), Trend.DOWN)

    def test_idle_then_active_is_up(self):
        self.assertIs(classify(3.0, 0.0), Trend.UP)
        self.assertIs(classify(0.0, 0.0), Trend.FLAT)


class TestTrackerHistory(unittest.TestCase):
    def build(self, pairs, depth=5, display=3):
        """Feed frames of (in_cumulative, out_cumulative, stamp)."""
        tracker = TrendTracker(depth=depth)
        for cin, cout, stamp in pairs:
            tracker.observe({K_IN: cin, K_OUT: cout}, stamp)
        return tracker, display

    def test_needs_two_frames(self):
        tracker, _ = self.build([(0.0, 0.0, 100.0)])
        self.assertEqual(tracker.intervals(3, IN), {})

    def test_returns_requested_number_oldest_first(self):
        tracker, display = self.build(
            [(0, 0, 100), (500, 100, 105), (1000, 200, 110), (1500, 300, 115), (2000, 400, 120), (2500, 500, 125)]
        )
        history = tracker.intervals(display, IN)[ROW]
        self.assertEqual(len(history), display)
        # constant 100/s, so every slot after the first is flat, and the first has a predecessor
        self.assertEqual([round(i.per_sec) for i in history], [100, 100, 100])
        self.assertEqual({i.trend for i in history}, {Trend.FLAT})

    def test_every_displayed_slot_has_a_predecessor(self):
        # the ring keeps one interval more than shown, so the leftmost is not left blank
        tracker, display = self.build([(0, 0, 100), (100, 0, 105), (200, 0, 110), (300, 0, 115), (900, 0, 120)])
        history = tracker.intervals(display, IN)[ROW]
        self.assertEqual(len(history), display)
        self.assertNotIn(Trend.UNKNOWN, [i.trend for i in history])

    def test_directions_across_a_rise_and_fall(self):
        tracker, display = self.build(
            [(0, 0, 100), (500, 100, 105), (1000, 200, 110), (1900, 300, 115), (1930, 400, 120)]
        )
        history = tracker.intervals(display, IN)[ROW]
        self.assertEqual([i.trend for i in history], [Trend.FLAT, Trend.UP, Trend.DOWN])

    def test_rate_normalises_by_elapsed_not_frame_count(self):
        # the 10s frames carry twice the tokens, so the rate is unchanged and must read flat
        tracker, display = self.build([(0, 0, 100), (500, 0, 105), (1000, 0, 110), (2000, 0, 120), (3000, 0, 130)])
        history = tracker.intervals(display, IN)[ROW]
        self.assertEqual({i.trend for i in history}, {Trend.FLAT})
        self.assertEqual([round(i.per_sec) for i in history], [100, 100, 100])

    def test_counter_reset_clears_history_rather_than_reading_as_down(self):
        tracker = TrendTracker(depth=5)
        tracker.observe({K_IN: 9000.0, K_OUT: 0.0}, 100.0)
        tracker.observe({K_IN: 10000.0, K_OUT: 0.0}, 105.0)
        tracker.observe({K_IN: 120.0, K_OUT: 0.0}, 110.0)  # gateway restarted
        self.assertEqual(tracker.frames_held(), 1)
        self.assertEqual(tracker.intervals(3, IN), {})

    def test_non_monotonic_stamp_is_ignored(self):
        tracker = TrendTracker()
        tracker.observe({K_IN: 10.0}, 100.0)
        tracker.observe({K_IN: 20.0}, 100.0)
        self.assertEqual(tracker.frames_held(), 1)

    def test_rows_are_tracked_separately(self):
        tracker = TrendTracker(depth=5)
        other = ("acme", "batch", "glm-5")
        for stamp, (a, b) in ((100, (0, 0)), (105, (500, 500)), (110, (1000, 600)),
                              (115, (1500, 700)), (120, (2500, 800))):
            tracker.observe({K_IN: a, (other[0], other[1], other[2], IN): b}, float(stamp))
        out = tracker.intervals(3, IN)
        self.assertEqual([round(i.per_sec) for i in out[ROW]], [100, 100, 200])
        self.assertEqual([round(i.per_sec) for i in out[other]], [20, 20, 20])

    def test_reset_empties_the_ring(self):
        tracker = TrendTracker()
        tracker.observe({K_IN: 0.0}, 100.0)
        tracker.observe({K_IN: 50.0}, 105.0)
        tracker.reset()
        self.assertEqual(tracker.frames_held(), 0)
        self.assertEqual(tracker.intervals(3, IN), {})

    def test_streams_are_independent_histories(self):
        tracker, display = self.build(
            [(0, 0, 100), (500, 100, 105), (1000, 200, 110), (1900, 300, 115), (1930, 1300, 120)]
        )
        # IN collapses while OUT surges on the same frame
        self.assertIs(tracker.intervals(display, IN)[ROW][-1].trend, Trend.DOWN)
        self.assertIs(tracker.intervals(display, OUT)[ROW][-1].trend, Trend.UP)


class TestCombine(unittest.TestCase):
    def test_disagreement_reads_flat(self):
        combined = combine(iv(10, 200, 5, Trend.UP), iv(4, 4, 5, Trend.DOWN))
        self.assertIs(combined.trend, Trend.FLAT)

    def test_single_mover_decides_not_set_ordering(self):
        # deterministic: IN up + OUT flat must always be up, whatever the set iteration order
        for _ in range(20):
            assert combine(iv(10, 200, 5, Trend.UP), iv(4, 4, 5, Trend.FLAT)).trend is Trend.UP
            assert combine(iv(4, 4, 5, Trend.FLAT), iv(10, 200, 5, Trend.DOWN)).trend is Trend.DOWN

    def test_unknown_dominates(self):
        assert combine(iv(1, 1, 5, Trend.UNKNOWN), iv(1, 1, 5, Trend.UP)).trend is Trend.UNKNOWN

    def test_sums_and_widest_span(self):
        combined = combine(iv(100, 20.0, 5.0, Trend.FLAT), iv(50, 10.0, 7.0, Trend.FLAT))
        assert (combined.tokens, combined.per_sec, combined.seconds) == (150, 30.0, 7.0)

    def test_nothing_to_combine(self):
        assert combine() is None
        assert combine(None, None) is None
        assert combine(iv(1, 1, 1, Trend.UP), None).trend is Trend.UP


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
