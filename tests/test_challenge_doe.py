from datetime import date

import numpy as np

from aios_track2.challenge_doe import (
    CHALLENGE_DIMENSIONS,
    FROZEN_CHALLENGE_SHA256,
    challenge_design_sha256,
    deterministic_spatial_groups,
    frozen_challenge_doe,
    schedule_role_names,
)
from aios_track2.challenge_schedule import scale_schedule_with_role_policies
from aios_track2.deck import Well


def test_frozen_challenge_design_is_preregistered_18d_40_8_16() -> None:
    rows = frozen_challenge_doe()
    assert len(rows) == 64
    assert CHALLENGE_DIMENSIONS == 18
    assert challenge_design_sha256() == FROZEN_CHALLENGE_SHA256
    assert sum(row.split == "train" for row in rows) == 40
    assert sum(row.split == "validation" for row in rows) == 8
    assert sum(row.split == "holdout" for row in rows) == 16
    assert np.allclose(rows[0].flat_vector(), 1.0)
    matrix = np.stack([row.flat_vector() for row in rows]).reshape(64, 6, 3)
    assert np.max(np.abs(np.diff(matrix, axis=2))) <= 0.120000000001
    assert np.min(matrix) >= 0.8
    assert np.max(matrix) <= 1.2


def test_spatial_grouping_is_deterministic_and_balanced() -> None:
    wells = [Well(f"W{i}", i % 5 + 1, i // 5 + 1, "OIL") for i in range(20)]
    first = deterministic_spatial_groups(wells, 4)
    second = deterministic_spatial_groups(list(reversed(wells)), 4)
    assert first == second
    counts = np.bincount(list(first.values()), minlength=4)
    assert counts.max() - counts.min() <= 1


def test_schedule_roles_allow_same_well_to_switch_role() -> None:
    text = """WCONPROD
 'W1' 'OPEN' 'LRAT' 3* 100 1* 120 /
 'P2' 'OPEN' 'LRAT' 3* 100 1* 120 /
/
WCONINJE
 'W1' 'WATER' 'OPEN' 'RATE' 50 1* 300 /
 'I2' 'WATER' 'OPEN' 'RATE' 50 1* 300 /
/
"""
    producers, injectors = schedule_role_names(text)
    assert producers == {"W1", "P2"}
    assert injectors == {"W1", "I2"}
    assert producers & injectors == {"W1"}


def test_role_specific_policy_can_scale_same_well_differently_by_role() -> None:
    source = """DATES
 1 JAN 2007 /
/
WCONPROD
 'W1' 'OPEN' 'LRAT' 3* 100 1* 120 /
/
WCONINJE
 'W1' 'WATER' 'OPEN' 'RATE' 50 1* 300 /
/
"""
    result = scale_schedule_with_role_policies(
        source,
        producer_well_groups={"W1": 0},
        injector_well_groups={"W1": 0},
        producer_group_nodes={0: (1.2,)},
        injector_group_nodes={0: (0.8,)},
        node_dates=(date(2007, 1, 1),),
        effective_from=date(2007, 1, 1),
    )
    assert "'W1' 'OPEN' 'LRAT' 3* 120.000000 1* 120" in result
    assert "'W1' 'WATER' 'OPEN' 'RATE' 40.000000 1* 300" in result


def test_role_specific_policy_preserves_pre_2007_history() -> None:
    source = """DATES
 1 JAN 2006 /
/
WCONPROD
 'W1' 'OPEN' 'LRAT' 3* 100 1* 120 /
/
DATES
 1 JAN 2007 /
/
WCONPROD
 'W1' 'OPEN' 'LRAT' 3* 100 1* 120 /
/
"""
    result = scale_schedule_with_role_policies(
        source,
        producer_well_groups={"W1": 0},
        injector_well_groups={},
        producer_group_nodes={0: (1.2,)},
        injector_group_nodes={},
        node_dates=(date(2007, 1, 1),),
        effective_from=date(2007, 1, 1),
    )
    before, after = result.split("DATES\n 1 JAN 2007 /", 1)
    assert "3* 100 1* 120 /" in before
    assert "3* 120.000000 1* 120" in after
