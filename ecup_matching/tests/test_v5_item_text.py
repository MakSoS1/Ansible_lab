from ecup_matching.ml.textnorm import normalize_item
from ecup_matching.ml.v5_item_text import serialize_item_v5


def test_v5_item_serializer_preserves_name_model_numbers_units_and_brand():
    item = normalize_item(
        1,
        "Смартфон Samsung Galaxy S24 SM-S921B 256GB",
        '{"Бренд":"Samsung","Модель":"SM-S921B","Память":"256 GB","Вес":"167 г","Цвет":"черный"}',
        "Электроника",
    )
    text = serialize_item_v5(item, max_chars=1200)

    assert "[NAME]" in text
    assert "samsung galaxy s24 sm-s921b 256gb" in text
    assert "[BRAND] samsung" in text
    assert "[MODEL]" in text and "sm-s921b" in text
    assert "[NUMERIC]" in text and "256" in text
    assert "[ATTR]" in text and "цвет=черный" in text


def test_v5_item_serializer_is_deterministic_and_hard_bounded_without_cutting_prefix_sections():
    attrs = {f"attr_{i}": "очень длинное значение " * 10 for i in range(30)}
    attrs["Бренд"] = "ACME"
    attrs["Модель"] = "ZX-9000"
    item = normalize_item(7, "ACME ZX-9000 2 кг", attrs, "Дом и сад")

    first = serialize_item_v5(item, max_chars=240)
    second = serialize_item_v5(item, max_chars=240)

    assert first == second
    assert len(first) <= 240
    assert first.startswith("[NAME]")
    assert "zx-9000" in first
    assert "[BRAND] acme" in first
