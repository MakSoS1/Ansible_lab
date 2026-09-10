from __future__ import annotations

import hashlib
import json

from aios_track2.real_doe import FROZEN_DESIGN_SHA256, frozen_real_doe


def _digest(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def test_frozen_real_doe_is_preregistered_before_targets_exist() -> None:
    scenarios = frozen_real_doe()
    rows = [scenario.as_dict() for scenario in scenarios]

    assert len(rows) == 32
    assert sum(row["split"] == "train" for row in rows) == 20
    assert sum(row["split"] == "validation" for row in rows) == 4
    assert sum(row["split"] == "holdout" for row in rows) == 8
    assert rows[0] == {
        "scenario_id": 0,
        "split": "train",
        "producer_2007": 1.0,
        "producer_2025": 1.0,
        "injector_2007": 1.0,
        "injector_2025": 1.0,
    }
    assert _digest(rows) == FROZEN_DESIGN_SHA256
    assert FROZEN_DESIGN_SHA256 == "571b00af32773c13df8dd4a9497f8096ad6588fd34dcc5fb73619fc042cb6b9a"


def test_frozen_real_doe_keeps_nonbaseline_controls_inside_declared_range() -> None:
    scenarios = frozen_real_doe()
    for scenario in scenarios[1:]:
        for value in (
            scenario.producer_2007,
            scenario.producer_2025,
            scenario.injector_2007,
            scenario.injector_2025,
        ):
            assert 0.8 <= value <= 1.2
