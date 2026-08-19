import json

from ecup_matching.attribute_schema_audit import inspect_attribute_payload


def test_attribute_schema_audit_counts_nested_lists_and_leaf_collisions():
    payload = {
        "technical": {"color": "black", "storage": "128 GB"},
        "appearance": {"color": "graphite"},
        "variants": [
            {"sku": "A", "power": "65 W"},
            {"sku": "B", "power": "45 W"},
        ],
        "tags": ["phone", "flagship"],
    }
    report = inspect_attribute_payload(json.dumps(payload, ensure_ascii=False))

    assert report["parse_success"] is True
    assert report["top_level_type"] == "dict"
    assert report["nested_dict_count"] >= 2
    assert report["list_str_count"] == 1
    assert report["list_dict_count"] == 1
    assert report["leaf_count"] >= 8
    assert report["leaf_key_collision_count"] >= 1
    assert "color" in report["colliding_leaf_keys"]


def test_attribute_schema_audit_handles_invalid_and_top_level_list():
    bad = inspect_attribute_payload("not json")
    assert bad["parse_success"] is False
    assert bad["top_level_type"] == "invalid"

    listed = inspect_attribute_payload('[{"a":1},{"a":2}]')
    assert listed["parse_success"] is True
    assert listed["top_level_type"] == "list"
    assert listed["list_dict_count"] >= 1
