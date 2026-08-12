from ecup_matching.ml.textnorm import normalize_item
from ecup_matching.ml.v7_item_text import serialize_item_v7


def test_fashion_identity_keeps_material_gender_and_season_in_front_context():
    item = normalize_item(
        101,
        "Женская зимняя куртка Nord размер 46 черная",
        {
            "Размер": "46",
            "Цвет": "черный",
            "Пол": "женский",
            "Материал": "полиэстер",
            "Состав": "полиэстер 100%",
            "Сезон": "зима",
            **{f"служебное число {i:02d}": str(100000 + i) for i in range(20)},
        },
        "Одежда",
    )
    text = serialize_item_v7(item, max_chars=300)
    assert "пол=женский" in text
    assert "материал=полиэстер" in text
    assert "сезон=зима" in text
    assert text.index("материал=полиэстер") < text.find("[NUMERIC]") if "[NUMERIC]" in text else True


def test_jewelry_identity_keeps_hallmark_and_stone_in_front_context():
    item = normalize_item(
        102,
        "Кольцо золотое с фианитом размер 17",
        {
            "Материал": "золото",
            "Проба": "585",
            "Вставка": "фианит",
            "Размер": "17",
            **{f"служебное число {i:02d}": str(200000 + i) for i in range(20)},
        },
        "Ювелирные изделия",
    )
    text = serialize_item_v7(item, max_chars=260)
    assert "материал=золото" in text
    assert "проба=585" in text
    assert "вставка=фианит" in text
    assert text.index("проба=585") < text.find("[NUMERIC]") if "[NUMERIC]" in text else True
