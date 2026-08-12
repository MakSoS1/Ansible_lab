from ecup_matching.ml.textnorm import normalize_item
from ecup_matching.ml.v7_item_text import serialize_item_v7


def test_v7_serializer_surfaces_category_before_attribute_sections():
    item = normalize_item(
        7,
        "Женские кроссовки Nord 39 размер",
        {"Размер": "39", "Материал": "кожа", "Цвет": "белый"},
        "Обувь",
    )
    text = serialize_item_v7(item, max_chars=240)
    assert "[CATEGORY] обувь" in text
    assert text.index("[CATEGORY]") < text.index("[IDENTITY]")
