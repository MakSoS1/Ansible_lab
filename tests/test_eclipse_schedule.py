from datetime import date

from aios_track2.eclipse_schedule import scale_schedule_text


def test_scale_schedule_text_updates_active_targets_and_preserves_limits() -> None:
    source = """WCONPROD
 'P1' 'OPEN' 'LRAT' 3* 400 1* 120 /
 'P2' 'OPEN' 'ORAT' 100 4* 90 /
/
WCONINJE
 'I1' 'WATER' 'OPEN' 'RATE' 250 1* 300 /
/
"""
    result = scale_schedule_text(
        source,
        producer_scale={"P1": 1.5, "P2": 0.8},
        injector_scale={"I1": 1.2},
        max_wlpr=500,
    )
    assert "'P1' 'OPEN' 'LRAT' 3* 500.000000 1* 120" in result
    assert "'P2' 'OPEN' 'ORAT' 80.000000 4* 90" in result
    assert "'I1' 'WATER' 'OPEN' 'RATE' 300.000000 1* 300" in result


def test_scale_schedule_text_leaves_unselected_wells_untouched() -> None:
    source = """WCONPROD
 'P1' 'OPEN' 'LRAT' 3* 200 1* 120 /
 'P2' 'OPEN' 'LRAT' 3* 250 1* 130 /
/
"""
    result = scale_schedule_text(source, producer_scale={"P1": 1.1}, injector_scale={})
    assert "'P1' 'OPEN' 'LRAT' 3* 220.000000 1* 120" in result
    assert "'P2' 'OPEN' 'LRAT' 3* 250 1* 130 /" in result


def test_scale_schedule_text_handles_compressed_default_items() -> None:
    source = """WCONPROD
 'P1' 'OPEN' 'LRAT' 3* 350 2* 170 /
/
"""
    result = scale_schedule_text(source, producer_scale={"P1": 0.5}, injector_scale={})
    assert "'P1' 'OPEN' 'LRAT' 3* 175.000000 2* 170" in result


def test_history_before_effective_date_is_byte_stable() -> None:
    source = """DATES
 1 JAN 2006 /
/
WCONPROD
 'P1' 'OPEN' 'LRAT' 3* 200 1* 120 /
/
DATES
 1 JAN 2007 /
/
WCONPROD
 'P1' 'OPEN' 'LRAT' 3* 200 1* 120 /
/
"""
    result = scale_schedule_text(
        source,
        producer_scale={"P1": 1.5},
        injector_scale={},
        effective_from=date(2007, 1, 1),
    )
    before, after = result.split("DATES\n 1 JAN 2007 /", 1)
    assert "3* 200 1* 120 /" in before
    assert "3* 300.000000 1* 120" in after
