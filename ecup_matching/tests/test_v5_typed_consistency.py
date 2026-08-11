import pytest

from ecup_matching.ml.textnorm import normalize_item
from ecup_matching.ml.v5_typed_consistency import typed_quantity_consistency


def _item(item_id: int, name: str):
    return normalize_item(item_id, name, "{}", "x")


def test_typed_quantity_consistency_is_symmetric_bounded_and_unit_canonical():
    a = _item(1, "SSD 128 GB charger 65 W")
    b = _item(2, "SSD 0.128 TB charger 0.065 kW")
    c = _item(3, "SSD 256 GB charger 65 W")

    assert typed_quantity_consistency(a, b) == pytest.approx(1.0)
    assert typed_quantity_consistency(b, a) == pytest.approx(1.0)
    assert typed_quantity_consistency(a, c) == pytest.approx(0.0)
    assert -1.0 <= typed_quantity_consistency(a, c) <= 1.0


def test_typed_quantity_consistency_penalizes_conflict_and_ignores_missing_dimensions():
    a = _item(1, "phone 128GB 5000mAh")
    conflict = _item(2, "phone 256GB 5000mAh")
    missing = _item(3, "phone black")

    assert typed_quantity_consistency(a, conflict) == pytest.approx(0.0)
    assert typed_quantity_consistency(a, missing) == pytest.approx(0.0)

    only_conflict = _item(4, "SSD 256 GB")
    assert typed_quantity_consistency(_item(5, "SSD 128GB"), only_conflict) == pytest.approx(-1.0)
