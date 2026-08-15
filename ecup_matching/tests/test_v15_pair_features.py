import json
import numpy as np

from ecup_matching.v15_fields import normalize_item_fields
from ecup_matching.v15_pair_features import PAIR_FEATURE_NAMES, build_pair_features


def _item(name, attrs):
    return normalize_item_fields(name, json.dumps(attrs, ensure_ascii=False), "Электроника")


def test_v15_pair_features_are_exactly_symmetric():
    a = _item("iPhone 15 Pro 128GB A3101", {"Бренд": "Apple", "Модель": "A3101", "Память": "128 ГБ"})
    b = _item("Apple iPhone 15 Pro 256GB A3102", {"Бренд": "Apple", "Модель": "A3102", "Память": "256 ГБ"})
    ab = build_pair_features(a, b)
    ba = build_pair_features(b, a)
    assert ab.shape == (len(PAIR_FEATURE_NAMES),)
    np.testing.assert_allclose(ab, ba, rtol=0, atol=0)


def test_v15_pair_features_surface_model_and_numeric_conflicts():
    a = _item("RTX 4070 12GB", {"Бренд": "NVIDIA", "Модель": "RTX4070", "Память": "12 GB"})
    b = _item("RTX 4070 Ti 16GB", {"Бренд": "NVIDIA", "Модель": "RTX4070TI", "Память": "16 GB"})
    features = dict(zip(PAIR_FEATURE_NAMES, build_pair_features(a, b)))
    assert features["brand_equal"] == 1.0
    assert features["model_exact"] == 0.0
    assert features["model_conflict"] == 1.0
    assert features["numeric_conflict_count"] >= 1.0


def test_v15_pair_features_identical_items_have_no_conflict():
    a = _item("Samsung SSD 990 PRO 2TB", {"Бренд": "Samsung", "Модель": "990 PRO", "Объем": "2 TB"})
    features = dict(zip(PAIR_FEATURE_NAMES, build_pair_features(a, a)))
    assert features["brand_equal"] == 1.0
    assert features["model_exact"] == 1.0
    assert features["model_conflict"] == 0.0
    assert features["numeric_conflict_count"] == 0.0
