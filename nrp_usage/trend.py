"""Rolling token production history for --watch.

``gen_ai_client_token_usage_sum`` is a cumulative counter, so a single delta against the
previous frame can only ever be "up". A meaningful increasing/decreasing signal has to
compare *production rates* between consecutive intervals.

The tracker therefore keeps a ring of recent frames and derives intervals from it, which is
what lets the team table show the last N frames side by side, oldest on the left.

Counters also reset when the gateway restarts. That is not a slowdown and must never be
rendered as one, so a reset interval is reported as unknown and the ring is cleared.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise

#: (team_id, token_alias, model, token_type)
SamplePair = tuple[str, str, str, str]
#: (team_id, token_alias, model)
RowPair = tuple[str, str, str]

INPUT, OUTPUT = "input", "output"


class Trend(Enum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"
    UNKNOWN = "unknown"  # no predecessor, or a counter reset broke the chain


@dataclass(frozen=True)
class Interval:
    """What one frame-to-frame interval produced, and how that compares with the one before."""

    tokens: float
    per_sec: float
    seconds: float
    trend: Trend

    def __bool__(self) -> bool:
        return True


NONE = Interval(0.0, 0.0, 0.0, Trend.UNKNOWN)


#: a rate within this fraction of the previous one is unchanged, so that ordinary scrape
#: jitter does not produce a stream of alternating arrows
FLAT_BAND = 0.10


def classify(rate: float, previous: float | None, flat_band: float = FLAT_BAND) -> Trend:
    """Up / down / flat for one interval against its predecessor."""
    if previous is None:
        return Trend.UNKNOWN
    if previous == 0 and rate == 0:
        return Trend.FLAT
    if previous == 0:
        return Trend.UP  # idle, then producing
    ratio = (rate - previous) / previous
    if ratio > flat_band:
        return Trend.UP
    if ratio < -flat_band:
        return Trend.DOWN
    return Trend.FLAT


@dataclass
class TrendTracker:
    """Keeps the last few sampled frames and derives per-interval production from them."""

    depth: int = 5  # frames retained: one more than the intervals derived, than displayed
    _ring: deque[tuple[float, dict[SamplePair, float]]] = field(default_factory=deque)

    def observe(self, samples: dict[SamplePair, float], stamp: float) -> None:
        """Record one frame. ``stamp`` must increase monotonically between calls."""
        if self._ring and stamp <= self._ring[-1][0]:
            return  # clock did not move: nothing sensible can be derived
        if self._is_reset(samples):
            self._ring.clear()  # a restart is not a slowdown
        self._ring.append((stamp, dict(samples)))
        while len(self._ring) > self.depth:
            self._ring.popleft()

    def _is_reset(self, samples: dict[SamplePair, float]) -> bool:
        if not self._ring:
            return False
        previous = self._ring[-1][1]
        return any(value < previous.get(key, value) for key, value in samples.items())

    def intervals(self, limit: int, token_type: str) -> dict[RowPair, list[Interval]]:
        """Per row, the last ``limit`` intervals ending now, oldest first.

        An extra interval is computed where available so the leftmost displayed column can
        still be compared against its predecessor.
        """
        out: dict[RowPair, list[Interval]] = {}
        if len(self._ring) < 2:
            return out
        frames = list(self._ring)
        rows: dict[RowPair, None] = {}
        for _stamp, values in frames:
            for (team, alias, model, kind) in values:
                if kind == token_type:
                    rows.setdefault((team, alias, model), None)
        for row in rows:
            derived: list[Interval] = []
            for (e_stamp, e_vals), (l_stamp, l_vals) in pairwise(frames):
                key_in = (row[0], row[1], row[2], token_type)
                if key_in not in e_vals or key_in not in l_vals:
                    continue
                seconds = l_stamp - e_stamp
                if seconds <= 0:
                    continue
                tokens = max(0.0, l_vals[key_in] - e_vals[key_in])
                derived.append(Interval(tokens, tokens / seconds, seconds, Trend.UNKNOWN))
            if not derived:
                continue
            for index, item in enumerate(derived):
                before = derived[index - 1].per_sec if index else None
                derived[index] = Interval(item.tokens, item.per_sec, item.seconds, classify(item.per_sec, before))
            out[row] = derived[-limit:]
        return out

    def frames_held(self) -> int:
        return len(self._ring)

    def reset(self) -> None:
        self._ring.clear()


def combine(*streams: Interval | None) -> Interval | None:
    """Fold intervals from separate streams (input, output) into one comparison."""
    parts = [p for p in streams if p is not None]
    if not parts:
        return None
    tokens = sum(p.tokens for p in parts)
    rate = sum(p.per_sec for p in parts)
    seconds = max(p.seconds for p in parts)
    votes = {p.trend for p in parts}
    if Trend.UNKNOWN in votes:
        trend = Trend.UNKNOWN
    else:
        movers = votes - {Trend.FLAT}
        # exactly one stream moving decides the pair; a genuine disagreement does not, and
        # picking next(iter(votes)) would let set ordering decide the arrow
        trend = next(iter(movers)) if len(movers) == 1 else Trend.FLAT
    return Interval(tokens, rate, seconds, trend)
