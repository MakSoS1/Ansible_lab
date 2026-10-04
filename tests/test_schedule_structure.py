from datetime import date

from aios_track2.schedule_structure import advance_by_tsteps, inspect_schedule_text


def test_inspect_schedule_structure_counts_dates_tsteps_and_modes() -> None:
    text = """DATES
 1 JAN 2007 /
 1 APR 2007 /
/
WCONPROD
 'P1' 'OPEN' 'LRAT' 3* 100 /
 'P2' 'OPEN' 'ORAT' 80 /
/
TSTEP
 2*30 15 /
/
WCONINJE
 'I1' 'WATER' 'OPEN' 'RATE' 200 /
/
"""
    result = inspect_schedule_text(text)
    assert result.explicit_dates == (date(2007, 1, 1), date(2007, 4, 1))
    assert result.tstep_days == (30.0, 30.0, 15.0)
    assert result.producer_blocks == 1
    assert result.producer_records == 2
    assert result.injector_records == 1
    assert result.producer_modes == ("LRAT", "ORAT")
    assert result.injector_modes == ("RATE",)


def test_advance_by_tsteps_is_deterministic() -> None:
    assert advance_by_tsteps(date(2007, 1, 1), (30.0, 30.0)) == (date(2007, 1, 31), date(2007, 3, 2))
