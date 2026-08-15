import json

from ecup_matching.v15_fields import normalize_item_fields
from ecup_matching.v15_serialization import serialize_pair


def test_v15_serialization_has_stable_field_markers_and_attribute_order():
    a = normalize_item_fields("Phone X", json.dumps({"z": "2", "a": "1"}), "Electronics")
    b = normalize_item_fields("Phone X", json.dumps({"a": "1", "z": "2"}), "Electronics")
    left, right = serialize_pair(a, b)
    assert left.startswith("[TITLE] phone x")
    assert "[CATEGORY] electronics" in left
    assert left.index("a=1") < left.index("z=2")
    assert left == right


def test_v15_serialization_does_not_emit_raw_malformed_json():
    a = normalize_item_fields("Widget", "{totally broken", "Tools")
    b = normalize_item_fields("Widget 2", "", "Tools")
    left, right = serialize_pair(a, b)
    assert "totally broken" not in left
    assert "[TITLE] widget" in left
    assert "[TITLE] widget 2" in right
