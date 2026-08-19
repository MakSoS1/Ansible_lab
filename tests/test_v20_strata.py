from ecup_matching.ml.textnorm import normalize_item
from ecup_matching.ml.v20_strata import classify_pair_stratum


def n(i, name, attrs="{}", category="Электроника"):
    return normalize_item(i, name, attrs, category)


def test_model_conflict_is_symmetric():
    a = n(1, "Samsung Galaxy S24 SM-S921B 256 GB")
    b = n(2, "Samsung Galaxy S24 Ultra SM-S928B 256 GB")
    ab = classify_pair_stratum(a, b)
    ba = classify_pair_stratum(b, a)
    assert ab.reason_code == ba.reason_code == "MODEL_CONFLICT"
    assert ab.difficulty == ba.difficulty


def test_capacity_conflict_is_detected():
    a = n(1, "Apple iPhone 15 Pro 256 GB")
    b = n(2, "Apple iPhone 15 Pro 512 GB")
    assert classify_pair_stratum(a, b).reason_code == "CAPACITY_CONFLICT"


def test_size_conflict_from_attributes():
    a = n(1, "Кроссовки Model X", '{"Размер":"42"}', "Обувь")
    b = n(2, "Кроссовки Model X", '{"Размер":"44"}', "Обувь")
    assert classify_pair_stratum(a, b).reason_code == "SIZE_CONFLICT"


def test_accessory_is_detected():
    a = n(1, "Apple AirPods Pro 2")
    b = n(2, "Чехол для Apple AirPods Pro 2")
    assert classify_pair_stratum(a, b).reason_code == "ACCESSORY"


def test_sparse_evidence_is_explicit():
    a = n(1, "товар", "{}", "Дом и сад")
    b = n(2, "товар", "{}", "Дом и сад")
    assert classify_pair_stratum(a, b).reason_code == "SPARSE_EVIDENCE"


def test_same_identity_gets_same_model_reason():
    a = n(1, "Sony WH-1000XM5 black", '{"Бренд":"Sony"}')
    b = n(2, "Наушники Sony WH1000XM5 черные", '{"brand":"Sony"}')
    assert classify_pair_stratum(a, b).reason_code == "SAME_MODEL"
