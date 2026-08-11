import numpy as np
import pandas as pd

from ecup_matching.ml.v5_explicit_attributes import (
    build_explicit_attribute_features,
    learn_explicit_attribute_keys,
)


def _items():
    return pd.DataFrame(
        {
            "id": list(range(8)),
            "name": ["товар"] * 8,
            "category": ["electronics"] * 8,
            "attributes": [
                '{"brand":"A","model":"M1","memory":"256","color":"black"}',
                '{"brand":"A","model":"M1","memory":"256","color":"black"}',
                '{"brand":"A","model":"M1","memory":"128","color":"black"}',
                '{"brand":"A","model":"M2","memory":"256","color":"black"}',
                '{"brand":"A","model":"M3","memory":"512","color":"white"}',
                '{"brand":"A","model":"M3","memory":"512","color":"white"}',
                '{"brand":"A","model":"M3","memory":"256","color":"white"}',
                '{"brand":"B","model":"Z9","memory":"512","color":"white"}',
            ],
        }
    )


def test_explicit_key_learning_is_train_only_and_selects_discriminative_keys():
    items = _items()
    train = pd.DataFrame(
        {
            "id1": [0, 4, 0, 0, 4, 4],
            "id2": [1, 5, 2, 3, 6, 7],
            "target": [1, 1, 0, 0, 0, 0],
            "category": ["electronics"] * 6,
        }
    )
    spec = learn_explicit_attribute_keys(items, train, max_keys_per_category=3, min_support=2)
    assert "electronics" in spec
    assert "memory" in spec["electronics"] or "model" in spec["electronics"]
    assert len(spec["electronics"]) <= 3


def test_explicit_pair_features_expose_equal_conflict_and_missing_per_selected_key():
    items = _items()
    pairs = pd.DataFrame(
        {
            "id1": [0, 0, 0],
            "id2": [1, 2, 3],
            "category": ["electronics"] * 3,
        }
    )
    spec = {"electronics": ["model", "memory"]}
    features = build_explicit_attribute_features(items, pairs, spec)
    assert np.isfinite(features.to_numpy(dtype=float)).all()
    assert features.loc[0, "attr_eq::model"] == 1.0
    assert features.loc[0, "attr_eq::memory"] == 1.0
    assert features.loc[1, "attr_conflict::memory"] == 1.0
    assert features.loc[2, "attr_conflict::model"] == 1.0


def test_explicit_feature_schema_is_fixed_by_train_spec_not_held_values():
    items = _items()
    spec = {"electronics": ["model"]}
    a = build_explicit_attribute_features(
        items,
        pd.DataFrame({"id1": [0], "id2": [1], "category": ["electronics"]}),
        spec,
    )
    b = build_explicit_attribute_features(
        items,
        pd.DataFrame({"id1": [6], "id2": [7], "category": ["electronics"]}),
        spec,
    )
    assert a.columns.tolist() == b.columns.tolist()


def test_explicit_attributes_treat_equivalent_storage_units_as_equal():
    items = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["ssd"] * 3,
            "category": ["electronics"] * 3,
            "attributes": [
                '{"storage":"128 GB"}',
                '{"storage":"0.128 TB"}',
                '{"storage":"256GB"}',
            ],
        }
    )
    pairs = pd.DataFrame(
        {
            "id1": [1, 1],
            "id2": [2, 3],
            "category": ["electronics", "electronics"],
        }
    )
    features = build_explicit_attribute_features(items, pairs, {"electronics": ["storage"]})
    assert features.loc[0, "attr_eq::storage"] == 1.0
    assert features.loc[0, "attr_conflict::storage"] == 0.0
    assert features.loc[1, "attr_eq::storage"] == 0.0
    assert features.loc[1, "attr_conflict::storage"] == 1.0
