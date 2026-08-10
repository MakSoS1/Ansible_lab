import json

import numpy as np
import pandas as pd

import ecup_matching.ml.features_v2 as features_v2
from ecup_matching.ml.features import FEATURE_NAMES, _pair_features, normalize_items
from ecup_matching.ml.features_v2 import FEATURE_NAMES_V2, build_pair_features_v2


def _items():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Apple iPhone 15 Pro 128GB Black", "Apple iPhone 15 Pro 256GB Black", "Apple iPhone 15 Pro 128GB Black"],
        "attributes": [
            json.dumps({"brand": "Apple", "memory": "128 GB", "color": "black"}),
            json.dumps({"brand": "Apple", "memory": "256 GB", "color": "black"}),
            json.dumps({"brand": "Apple", "memory": "128 GB", "color": "black"}),
        ],
        "category": ["Electronics"] * 3,
    })


def test_v2_features_include_2024_transfer_signals():
    pairs = pd.DataFrame({"id1": [1, 1], "id2": [2, 3]})
    importance = {"electronics": {"memory": 5.0, "color": 0.2, "brand": 1.0}}
    out = build_pair_features_v2(_items(), pairs, attribute_importance=importance)
    assert list(out.columns) == list(FEATURE_NAMES_V2)
    assert out.loc[0, "brand_match"] == 1.0
    assert out.loc[0, "brand_conflict"] == 0.0
    assert out.loc[0, "critical_conflict_count"] >= 1.0
    assert out.loc[0, "weighted_attr_conflict"] > out.loc[1, "weighted_attr_conflict"]
    assert out.loc[0, "hard_negative_score"] > out.loc[1, "hard_negative_score"]


def test_v2_features_are_symmetric():
    items = _items(); importance = {"electronics": {"memory": 5.0, "color": 0.2, "brand": 1.0}}
    forward = build_pair_features_v2(items, pd.DataFrame({"id1": [1], "id2": [2]}), attribute_importance=importance)
    reverse = build_pair_features_v2(items, pd.DataFrame({"id1": [2], "id2": [1]}), attribute_importance=importance)
    pd.testing.assert_frame_equal(forward, reverse)


def test_single_pass_symmetry_matches_old_bidirectional_reference(monkeypatch):
    cache = normalize_items(_items()); a, b = cache[1], cache[2]
    ab = _pair_features(a, b); ba = _pair_features(b, a)
    expected = {
        name: ((str(ab["category"]) if str(ab["category"]) == str(ba["category"]) else min(str(ab["category"]), str(ba["category"]))) if name == "category" else (float(ab[name]) + float(ba[name])) / 2.0)
        for name in FEATURE_NAMES
    }
    original = features_v2._pair_features; calls = 0
    def counted(left, right):
        nonlocal calls; calls += 1; return original(left, right)
    monkeypatch.setattr(features_v2, "_pair_features", counted)
    actual = features_v2._symmetric_base_features(a, b)
    assert calls == 1
    assert actual["category"] == expected["category"]
    for name in FEATURE_NAMES:
        if name != "category": assert np.isclose(float(actual[name]), float(expected[name]), rtol=0.0, atol=1e-15), name


def test_unequal_name_lengths_do_not_repeat_reverse_partial_ratio(monkeypatch):
    items = pd.DataFrame({
        "id": [1, 2],
        "name": ["short phone 128", "a much longer short phone 128 black edition"],
        "attributes": ["{}", "{}"],
        "category": ["electronics", "electronics"],
    })
    cache = normalize_items(items); a, b = cache[1], cache[2]
    reference_ab = _pair_features(a, b); reference_ba = _pair_features(b, a)
    expected = (float(reference_ab["fuzz_partial_ratio"]) + float(reference_ba["fuzz_partial_ratio"])) / 2.0
    assert reference_ab["fuzz_partial_ratio"] == reference_ba["fuzz_partial_ratio"]
    original = features_v2._partial_ratio; calls = 0
    def counted(left, right):
        nonlocal calls; calls += 1; return original(left, right)
    monkeypatch.setattr(features_v2, "_partial_ratio", counted)
    actual = features_v2._symmetric_base_features(a, b)
    assert calls == 0
    assert np.isclose(float(actual["fuzz_partial_ratio"]), expected, rtol=0.0, atol=1e-15)
