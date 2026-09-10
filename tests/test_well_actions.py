from datetime import date

from aios_track2.well_actions import WellActionPlan, apply_well_actions, cyclic_factor


HISTORY_AND_FORECAST = """DATES
 1 JAN 2006 /
/
WCONPROD
 'P1' 'OPEN' 'LRAT' 3* 200 1* 120 /
 'P2' 'OPEN' 'LRAT' 3* 150 1* 120 /
/
WCONINJE
 'I1' 'WATER' 'OPEN' 'RATE' 100 1* 300 /
/
DATES
 1 JAN 2007 /
/
WCONPROD
 'P1' 'OPEN' 'LRAT' 3* 200 1* 120 /
 'P2' 'OPEN' 'LRAT' 3* 150 1* 120 /
/
WCONINJE
 'I1' 'WATER' 'OPEN' 'RATE' 100 1* 300 /
/
DATES
 1 JUL 2007 /
/
WCONPROD
 'P1' 'OPEN' 'LRAT' 3* 200 1* 120 /
 'P2' 'OPEN' 'LRAT' 3* 150 1* 120 /
/
WCONINJE
 'I1' 'WATER' 'OPEN' 'RATE' 100 1* 300 /
/
"""


def test_shut_in_changes_status_after_2007_only() -> None:
    result = apply_well_actions(
        HISTORY_AND_FORECAST,
        WellActionPlan(shut_wells=("P2",)),
        effective_from=date(2007, 1, 1),
    )
    before, after = result.split("DATES\n 1 JAN 2007 /", 1)
    assert "'P2' 'OPEN' 'LRAT' 3* 150 1* 120 /" in before
    assert "'P2' 'SHUT' 'LRAT' 3* 150 1* 120" in after
    assert "'P1' 'OPEN' 'LRAT' 3* 200 1* 120" in after


def test_producer_to_injector_conversion_emits_wconinje_and_shuts_oil() -> None:
    result = apply_well_actions(
        HISTORY_AND_FORECAST,
        WellActionPlan(convert_to_injector={"P2": 80.0}),
        effective_from=date(2007, 1, 1),
    )
    before, after = result.split("DATES\n 1 JAN 2007 /", 1)
    assert "'P2' 'OPEN' 'LRAT' 3* 150 1* 120 /" in before
    assert "'P2' 'WATER'" not in before
    assert "'P2' 'SHUT' 'LRAT'" in after
    assert "'P2' 'WATER' 'OPEN' 'RATE' 80.000000" in after
    assert after.count("'P2' 'WATER' 'OPEN' 'RATE' 80.000000") == 2


def test_cyclic_injection_alternates_by_semester_without_touching_history() -> None:
    result = apply_well_actions(
        HISTORY_AND_FORECAST,
        WellActionPlan(
            cyclic_injectors=("I1",),
            cyclic_high=1.2,
            cyclic_low=0.8,
            cyclic_period_months=6,
            cyclic_origin=date(2007, 1, 1),
        ),
        effective_from=date(2007, 1, 1),
    )
    before, after = result.split("DATES\n 1 JAN 2007 /", 1)
    assert "'I1' 'WATER' 'OPEN' 'RATE' 100 1* 300 /" in before
    assert "'I1' 'WATER' 'OPEN' 'RATE' 120.000000" in after
    assert "'I1' 'WATER' 'OPEN' 'RATE' 80.000000" in after


def test_well_actions_keep_pre_2007_bytes_identical() -> None:
    result = apply_well_actions(
        HISTORY_AND_FORECAST,
        WellActionPlan(shut_wells=("P1",), convert_to_injector={"P2": 90.0}, cyclic_injectors=("I1",)),
        effective_from=date(2007, 1, 1),
    )
    before_src, _ = HISTORY_AND_FORECAST.split("DATES\n 1 JAN 2007 /", 1)
    before_out, _ = result.split("DATES\n 1 JAN 2007 /", 1)
    assert before_src == before_out


def test_cyclic_factor_period() -> None:
    assert cyclic_factor(date(2007, 1, 1), high=1.2, low=0.8, period_months=6, origin=date(2007, 1, 1)) == 1.2
    assert cyclic_factor(date(2007, 7, 1), high=1.2, low=0.8, period_months=6, origin=date(2007, 1, 1)) == 0.8
    assert cyclic_factor(date(2008, 1, 1), high=1.2, low=0.8, period_months=6, origin=date(2007, 1, 1)) == 1.2
