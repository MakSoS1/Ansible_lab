import json

from ecup_matching.v15_fields import normalize_item_fields


def test_v15_fields_parse_and_normalize_deterministically():
    attrs = json.dumps({"Память": "256 ГБ", "Бренд": "Apple", "Модель": "A3102", "Цвет": "Черный"}, ensure_ascii=False)
    item = normalize_item_fields("Apple iPhone 15 Pro 256GB A3102", attrs, "Электроника")
    assert item.title == "apple iphone 15 pro 256gb a3102"
    assert item.category == "электроника"
    assert item.brand == "apple"
    assert "a3102" in item.model_tokens
    assert "256" in item.numeric_tokens
    assert tuple(k for k, _ in item.attributes) == tuple(sorted(k for k, _ in item.attributes))


def test_v15_fields_malformed_json_is_safe_and_stable():
    item = normalize_item_fields("  RTX 4070 Ti  ", "not-json", "Видеокарты")
    assert item.title == "rtx 4070 ti"
    assert item.attributes == ()
    assert item.raw_attributes_parse_ok is False
    assert "4070" in item.numeric_tokens


def test_v15_fields_extracts_conservative_units_without_inventing_values():
    attrs = json.dumps({"Объем": "0.5 л", "Вес": "500 г", "Размер": "42"}, ensure_ascii=False)
    item = normalize_item_fields("Товар", attrs, "Дом")
    values = dict(item.attributes)
    assert values["вес"] == "500 г"
    assert values["объем"] == "0.5 л"
    assert "500" in item.numeric_tokens
    assert "0.5" in item.numeric_tokens
