"""A long phase must never be able to go silent.

A production run looked like an idle GPU because a multi-minute phase emitted
one line and then nothing. Count-based intervals alone cannot prevent that:
if each unit is slow, the next count threshold may be minutes away.
"""

from __future__ import annotations

import json

import pytest

from ecup_matching.ml.progress import ProgressReporter, format_eta


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _reporter(**kwargs):
    lines: list[dict] = []
    clock = _Clock()
    defaults = dict(
        phase="serialize-items",
        total=1000,
        every_units=100,
        every_seconds=30.0,
        emit=lambda text, flush=False: lines.append(json.loads(text)),
        clock=clock,
    )
    defaults.update(kwargs)
    return ProgressReporter(**defaults), lines, clock


def test_time_bound_fires_even_when_almost_nothing_completed():
    """The case that made a working job look dead."""
    reporter, lines, clock = _reporter()
    assert reporter.update(1) is False
    clock.advance(31.0)
    assert reporter.update(2) is True
    assert lines[-1]["done"] == 2
    assert lines[-1]["phase"] == "serialize-items"


def test_unit_bound_fires_before_the_time_bound_when_work_is_fast():
    reporter, lines, clock = _reporter()
    clock.advance(0.5)
    assert reporter.update(100) is True
    assert lines[-1]["done"] == 100


def test_every_tick_carries_the_fields_needed_to_diagnose_a_stall():
    reporter, lines, clock = _reporter()
    clock.advance(10.0)
    reporter.update(250)
    payload = lines[-1]
    for field in (
        "phase",
        "done",
        "total",
        "percent",
        "elapsed_seconds",
        "units_per_second",
        "recent_units_per_second",
        "eta",
        "rss_bytes",
        "peak_rss_bytes",
    ):
        assert field in payload, f"{field} missing from progress payload"
    assert payload["percent"] == pytest.approx(25.0)


def test_eta_uses_recent_throughput_not_the_lifetime_average():
    reporter, lines, clock = _reporter(total=1000, every_units=100)
    clock.advance(100.0)
    reporter.update(100)  # slow start: 1 unit/s
    clock.advance(1.0)
    reporter.update(200)  # fast now: 100 units/s
    assert lines[-1]["recent_units_per_second"] == pytest.approx(100.0)
    # 800 remaining at the recent rate, not at the 2 units/s lifetime average.
    assert lines[-1]["eta_seconds"] == pytest.approx(8.0, abs=0.5)


def test_finish_always_emits_a_final_line():
    reporter, lines, clock = _reporter()
    clock.advance(1.0)
    reporter.finish()
    assert lines[-1]["final"] is True
    assert lines[-1]["done"] == 1000


def test_reporter_rejects_a_configuration_that_could_stay_silent():
    with pytest.raises(ValueError, match="every_units or every_seconds"):
        ProgressReporter("x", 10, every_units=0, every_seconds=0)
    with pytest.raises(ValueError, match="phase"):
        ProgressReporter("", 10)
    with pytest.raises(ValueError, match="total"):
        ProgressReporter("x", -1)


def test_zero_total_does_not_divide_by_zero():
    reporter, lines, clock = _reporter(total=0)
    reporter.finish(0)
    assert lines[-1]["percent"] == 100.0


def test_eta_formatting_is_human_readable():
    assert format_eta(45) == "45s"
    assert format_eta(125) == "2m05s"
    assert format_eta(7325) == "2h02m05s"
    assert format_eta(float("inf")) == "unknown"
    assert format_eta(-1) == "unknown"
