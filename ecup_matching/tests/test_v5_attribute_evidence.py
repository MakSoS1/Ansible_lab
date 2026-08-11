import numpy as np
import pandas as pd

from ecup_matching.ml.v5_attribute_evidence import (
    build_attribute_evidence_features,
    fit_attribute_evidence,
)


def _items():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["p"] * 6,
            "category": ["electronics"] * 6,
            "attributes": [
                '{"brand":"A","model":"M1","memory":"256 GB","color":"black"}',
                '{"brand":"A","model":"M1","memory":"256 GB","color":"black"}',
                '{"brand":"A","model":"M1","memory":"128 GB","color":"black"}',
                '{"brand":"A","model":"M2","memory":"256 GB","color":"black"}',
                '{"brand":"A","model":"M3","memory":"512 GB","color":"white"}',
                '{"brand":"A","model":"M3","memory":"512 GB","color":"white"}',
            ],
        }
    )


def test_attribute_evidence_learns_key_specific_match_and_conflict_signal():
    items = _items()
    train = pd.DataFrame(
        {
            "id1": [1, 5, 1, 1, 2, 2],
            "id2": [2, 6, 3, 4, 3, 4],
            "target": [1, 1, 0, 0, 0, 0],
        }
    )
    evidence = fit_attribute_evidence(items, train, min_support=2, smoothing=1.0)
    pairs = pd.DataFrame({"id1": [1, 1, 1], "id2": [2, 3, 4]})
    features = build_attribute_evidence_features(items, pairs, evidence)

    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[0, "attr_evidence_sum"] > features.loc[1, "attr_evidence_sum"]
    assert features.loc[0, "attr_evidence_sum"] > features.loc[2, "attr_evidence_sum"]
    assert features.loc[1, "attr_evidence_negative"] < 0
    assert features.loc[2, "attr_evidence_negative"] < 0


def test_attribute_evidence_is_symmetric_and_handles_unseen_combinations():
    items = _items()
    train = pd.DataFrame({"id1": [1, 1, 5], "id2": [2, 3, 6], "target": [1, 0, 1]})
    evidence = fit_attribute_evidence(items, train, min_support=1, smoothing=1.0)
    forward = pd.DataFrame({"id1": [3, 4], "id2": [4, 6]})
    reverse = pd.DataFrame({"id1": [4, 6], "id2": [3, 4]})

    a = build_attribute_evidence_features(items, forward, evidence)
    b = build_attribute_evidence_features(items, reverse, evidence)
    assert np.allclose(a.to_numpy(), b.to_numpy())
