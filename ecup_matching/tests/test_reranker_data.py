import json

import pandas as pd

from ecup_matching.ml.reranker_data import (
    build_reranker_examples,
    serialize_item_text,
    serialize_pair,
)
from ecup_matching.ml.textnorm import normalize_item


def _item(i, name, attrs, category="electronics"):
    return normalize_item(i, name, json.dumps(attrs, ensure_ascii=False), category)


def test_item_serialization_orders_category_important_attributes_first():
    item = _item(1, "Apple iPhone 15 Pro", {"color": "black", "memory": "128 GB", "brand": "Apple"})
    importance = {"electronics": {"memory": 5.0, "brand": 2.0, "color": 0.1}}
    text = serialize_item_text(item, importance, max_attrs=3, max_chars=500)
    assert text.startswith("Apple iphone 15 pro".lower())
    assert text.index("memory=128 gb") < text.index("brand=apple") < text.index("color=black")


def test_pair_serialization_is_symmetric_by_item_id():
    a = _item(10, "Phone 128", {"memory": "128"})
    b = _item(5, "Phone 256", {"memory": "256"})
    importance = {"electronics": {"memory": 2.0}}
    ab = serialize_pair(a, b, importance)
    ba = serialize_pair(b, a, importance)
    assert ab == ba
    assert ab[0].startswith("phone 256")
    assert ab[1].startswith("phone 128")


def test_item_serialization_is_deterministically_truncated():
    attrs = {f"k{i}": "x" * 50 for i in range(20)}
    item = _item(1, "Product", attrs)
    a = serialize_item_text(item, {}, max_attrs=20, max_chars=120)
    b = serialize_item_text(item, {}, max_attrs=20, max_chars=120)
    assert a == b
    assert len(a) <= 120


def test_build_examples_preserves_soft_targets_and_weights():
    items = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["A", "B"],
            "attributes": ['{"brand":"x"}', '{"brand":"x"}'],
            "category": ["cat", "cat"],
        }
    )
    pairs = pd.DataFrame(
        {"id1": [1], "id2": [2], "target": [0.87], "sample_weight": [0.6], "category": ["cat"]}
    )
    out = build_reranker_examples(items, pairs, attribute_importance={})
    assert out.loc[0, "target"] == 0.87
    assert out.loc[0, "sample_weight"] == 0.6
    assert out.loc[0, "category"] == "cat"
    assert out.loc[0, "id1"] == 1 and out.loc[0, "id2"] == 2
    assert out.loc[0, "text_a"] and out.loc[0, "text_b"]
