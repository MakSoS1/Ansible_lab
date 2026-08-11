import json

import numpy as np
import pandas as pd

from ecup_matching.ml.features import FEATURE_NAMES, build_pair_features
from ecup_matching.ml.textnorm import canonical_attribute_value, extract_quantities, normalize_item


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


def test_typed_quantities_capture_storage_power_battery_frequency_and_diagonal():
    quantities = extract_quantities(
        "phone 128GB 5000mAh charger 65W 20V display 2.4GHz 15.6 inch"
    )
    assert ("storage_bytes", 128_000_000_000.0) in quantities
    assert ("battery_mah", 5000.0) in quantities
    assert ("power_w", 65.0) in quantities
    assert ("voltage_v", 20.0) in quantities
    assert ("frequency_hz", 2_400_000_000.0) in quantities
    assert ("diagonal_in", 15.6) in quantities


def test_typed_quantities_canonicalize_equivalent_units_and_detect_storage_conflict():
    left = normalize_item(1, "SSD 128 GB", '{"storage":"128 GB"}', "electronics")
    same = normalize_item(2, "SSD 0.128 TB", '{"storage":"0.128 TB"}', "electronics")
    different = normalize_item(3, "SSD 256GB", '{"storage":"256GB"}', "electronics")

    assert ("storage_bytes", 128_000_000_000.0) in left.quantities
    assert ("storage_bytes", 128_000_000_000.0) in same.quantities
    assert left.quantities & same.quantities
    assert left.quantities != different.quantities

    items = pd.DataFrame(
        [
            {"id": 1, "name": "SSD 128 GB", "attributes": "{}", "category": "electronics"},
            {"id": 2, "name": "SSD 0.128 TB", "attributes": "{}", "category": "electronics"},
            {"id": 3, "name": "SSD 256GB", "attributes": "{}", "category": "electronics"},
        ]
    )
    same_features = build_pair_features(items, pd.DataFrame([{"id1": 1, "id2": 2}])).iloc[0]
    conflict_features = build_pair_features(items, pd.DataFrame([{"id1": 1, "id2": 3}])).iloc[0]
    assert same_features["quantity_conflict"] == 0.0
    assert conflict_features["quantity_conflict"] == 1.0


def test_canonical_attribute_value_preserves_context_but_normalizes_units():
    assert canonical_attribute_value("128 GB") == canonical_attribute_value("0.128 TB")
    assert canonical_attribute_value("black 128GB") != canonical_attribute_value("black 256 GB")
    assert canonical_attribute_value("charger 0.065 kW") == canonical_attribute_value("charger 65 W")


def test_voltage_parser_avoids_ambiguous_russian_two_in_one_phrase():
    assert not any(dim == "voltage_v" for dim, _ in extract_quantities("устройство 2 в 1"))
    assert ("voltage_v", 220.0) in extract_quantities("adapter 220V")
