import pytest

from ecup_matching.ml.textnorm import normalize_item
from ecup_matching.ml.v5_model_code_consistency import model_code_consistency


def _item(item_id: int, name: str):
    return normalize_item(item_id, name, "{}", "x")


def test_model_code_consistency_is_symmetric_and_detects_match_conflict_missing():
    a = _item(1, "Samsung SM-S921B 128GB")
    same = _item(2, "case for SM S921B")
    conflict = _item(3, "Samsung SM-S926B 128GB")
    missing = _item(4, "Samsung smartphone black")

    assert model_code_consistency(a, same) == pytest.approx(1.0)
    assert model_code_consistency(same, a) == pytest.approx(1.0)
    assert model_code_consistency(a, conflict) == pytest.approx(-1.0)
    assert model_code_consistency(a, missing) == pytest.approx(0.0)


def test_storage_capacity_is_not_mistaken_for_model_code():
    a = _item(1, "SSD 128GB")
    b = _item(2, "SSD 256GB")
    assert not a.model_codes
    assert not b.model_codes
    assert model_code_consistency(a, b) == pytest.approx(0.0)
