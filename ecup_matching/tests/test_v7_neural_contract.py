import pandas as pd
import pytest

from ecup_matching.ml.textnorm import normalize_item
from ecup_matching.ml.v7_item_text import serialize_item_v7
from ecup_matching.ml.v7_teacher_contract import (
    V7TeacherConfig,
    filter_forbidden_weak_pairs,
    required_optimizer_steps,
    validate_v7_teacher_config,
)


def _phone(memory: str = "128 GB"):
    return normalize_item(
        1,
        "Смартфон Samsung Galaxy S24 SM-S921B 128GB 6.2 дюйма",
        {
            "Бренд": "Samsung",
            "Модель": "SM-S921B",
            "Встроенная память": memory,
            "Диагональ": "6.2 дюйма",
            "Емкость аккумулятора": "4000 mAh",
            "Цвет": "черный",
            **{f"служебный параметр {i:02d}": f"значение {i} 999999" for i in range(30)},
        },
        "Электроника",
    )


def test_identity_first_serializer_keeps_canonical_identity_before_residual_tail():
    text = serialize_item_v7(_phone(), max_chars=300)

    assert text.startswith("[NAME]")
    assert "[BRAND] samsung" in text
    assert "[MODEL]" in text and "sms921b" in text
    assert "storage_bytes_128000000000" in text
    assert "diagonal_in_6.2" in text
    assert "battery_mah_4000" in text
    assert text.index("storage_bytes_128000000000") < text.find("[RESIDUAL]", len(text)) if "[RESIDUAL]" in text else True
    assert len(text) <= 300


def test_identity_first_serializer_canonicalizes_equivalent_storage_units():
    gb = serialize_item_v7(_phone("128 GB"), max_chars=360)
    tb = serialize_item_v7(_phone("0.128 TB"), max_chars=360)

    assert "storage_bytes_128000000000" in gb
    assert "storage_bytes_128000000000" in tb


def test_identity_first_serializer_is_deterministic_and_does_not_cut_identity_packet():
    first = serialize_item_v7(_phone(), max_chars=220)
    second = serialize_item_v7(_phone(), max_chars=220)

    assert first == second
    assert len(first) <= 220
    assert "[MODEL] sms921b" in first
    assert "storage_bytes_128000000000" in first


def test_forbidden_weak_pair_filter_removes_either_endpoint_and_reports_it():
    weak = pd.DataFrame(
        {
            "id1": [1, 2, 3, 4],
            "id2": [10, 20, 30, 40],
            "target": [1.0, 0.0, 1.0, 0.0],
        }
    )
    kept, report = filter_forbidden_weak_pairs(weak, forbidden_item_ids={2, 30})

    assert list(zip(kept.id1, kept.id2)) == [(1, 10), (4, 40)]
    assert report == {"input_rows": 4, "removed_rows": 2, "kept_rows": 2}
    assert not ((set(kept.id1) | set(kept.id2)) & {2, 30})


def test_v7_training_contract_requires_full_context_and_nontrivial_exposure():
    legacy = V7TeacherConfig(max_length=128, curriculum_rows=120_000, effective_batch_size=32, epochs=1.0, max_steps=800)
    with pytest.raises(ValueError, match="max_length"):
        validate_v7_teacher_config(legacy)

    too_short = V7TeacherConfig(max_length=256, curriculum_rows=120_000, effective_batch_size=32, epochs=1.0, max_steps=800)
    with pytest.raises(ValueError, match="optimizer steps"):
        validate_v7_teacher_config(too_short)

    valid = V7TeacherConfig(max_length=256, curriculum_rows=240_000, effective_batch_size=32, epochs=1.0, max_steps=None)
    checked = validate_v7_teacher_config(valid)
    assert checked.max_length == 256
    assert required_optimizer_steps(checked) == 7500
