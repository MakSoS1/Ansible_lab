from datetime import date

from aios_track2.schedule import Control, Schedule, parse_schedule_inc, project_schedule, write_schedule_text


def test_wlpr_above_limit_is_rejected() -> None:
    schedule = Schedule(controls=(Control(date=date(2007, 1, 1), well="P1", wlpr=501.0),))
    result = project_schedule(schedule)
    assert result.accepted is False
    assert result.violations[0].code == "WLPR_LIMIT"


def test_schedule_text_is_stable() -> None:
    schedule = Schedule(controls=(Control(date=date(2007, 1, 1), well="P1", status="OPEN", wlpr=250.0),))
    assert write_schedule_text(schedule) == "DATES\n  1 JAN 2007 /\n/\nWCONPROD\n  'P1' 'OPEN' 'LRAT' 1* 250.000 /\n/\n"


def test_schedule_round_trip() -> None:
    schedule = Schedule(controls=(Control(date=date(2007, 1, 1), well="P1", status="OPEN", wlpr=250.0),))
    assert parse_schedule_inc(write_schedule_text(schedule)) == schedule
