import json

import pandas as pd

from ecup_matching.ml.category_attrs import (
    brand_from_item,
    learn_attribute_importance,
    weighted_attribute_stats,
)
from ecup_matching.ml.textnorm import normalize_item


def _item(i, name, attrs, category="electronics"):
    return normalize_item(i, name, json.dumps(attrs, ensure_ascii=False), category)


def test_brand_aliases_are_normalized():
    assert brand_from_item(_item(1, "Phone", {"Бренд": "Samsung"})) == "samsung"
    assert brand_from_item(_item(2, "Phone", {"brand": "Apple"})) == "apple"
    assert brand_from_item(_item(3, "Phone", {"Производитель": "Xiaomi"})) == "xiaomi"


def test_weighted_attribute_stats_emphasize_important_conflict():
    a = _item(1, "Phone X 128", {"brand": "A", "memory": "128 gb", "color": "black"})
    b = _item(2, "Phone X 256", {"brand": "A", "memory": "256 gb", "color": "black"})
    weights = {"electronics": {"memory": 5.0, "color": 0.2, "brand": 1.0}}
    stats = weighted_attribute_stats(a, b, weights)
    assert stats["weighted_attr_conflict"] > stats["weighted_attr_agreement"] * 0.5
    assert stats["brand_match"] == 1.0
    assert stats["brand_conflict"] == 0.0


def test_learn_attribute_importance_uses_training_labels_and_category():
    items = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["a"] * 6,
            "attributes": [
                '{"memory":"128","color":"black"}',
                '{"memory":"128","color":"white"}',
                '{"memory":"256","color":"black"}',
                '{"memory":"128","color":"black"}',
                '{"memory":"256","color":"white"}',
                '{"memory":"512","color":"white"}',
            ],
            "category": ["electronics"] * 6,
        }
    )
    pairs = pd.DataFrame(
        {
            "id1": [1, 1, 3, 5],
            "id2": [2, 3, 4, 6],
            "target": [1, 0, 0, 0],
            "category": ["electronics"] * 4,
        }
    )
    weights = learn_attribute_importance(items, pairs, min_support=2)
    assert "electronics" in weights
    assert weights["electronics"]["memory"] > weights["electronics"]["color"]
