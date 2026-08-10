import json

import numpy as np
import pandas as pd

from ecup_matching.ml.features import FEATURE_NAMES, build_pair_features


def _items():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": [
                "Смартфон Brand A15 128GB",
                "Смартфон Brand A15 128 GB",
                "Смартфон Brand A16 128GB",
                "Шампунь 500 мл",
                "Шампунь 1 л",
                "Смартфон Brand A15 128GB",
            ],
            "attributes": [
                json.dumps({"brand": "Brand", "memory": "128 GB"}, ensure_ascii=False),
                json.dumps({"memory": "128 GB", "brand": "Brand"}, ensure_ascii=False),
                json.dumps({"brand": "Brand", "memory": "128 GB"}, ensure_ascii=False),
                json.dumps({"volume": "500 ml"}),
                json.dumps({"volume": "1 l"}),
                json.dumps({"memory": "128 GB", "brand": "Brand"}, ensure_ascii=False),
            ],
            "category": ["phones", "phones", "phones", "care", "care", "phones"],
        }
    )


def test_features_capture_exact_equivalence_model_and_quantity_conflicts():
    pairs = pd.DataFrame({"id1": [1, 1, 4, 1], "id2": [6, 3, 5, 2]})
    x = build_pair_features(_items(), pairs)

    assert list(x.columns) == list(FEATURE_NAMES)
    assert len(x) == 4
    assert np.isfinite(x.select_dtypes(include=["number"]).to_numpy()).all()

    exact, model_conflict, quantity_conflict, reordered = [x.iloc[i] for i in range(4)]
    assert exact["name_exact"] == 1.0
    assert exact["attr_value_agreement"] == 1.0
    assert model_conflict["model_code_conflict"] == 1.0
    assert quantity_conflict["quantity_conflict"] == 1.0
    assert reordered["attr_value_agreement"] == 1.0
    assert reordered["fuzz_ratio"] > 0.9


def test_feature_output_preserves_pair_order_and_category():
    pairs = pd.DataFrame({"id1": [4, 1], "id2": [5, 2]})
    x = build_pair_features(_items(), pairs)
    assert x["category"].tolist() == ["care", "phones"]
