from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class _Norm:
    item_id: int
    category: str
    name: str
    attributes: str


class _TextNorm:
    @staticmethod
    def normalize_item(item_id, name, attributes, category):
        return _Norm(int(item_id), str(category), str(name), repr(attributes))


class _ItemText:
    @staticmethod
    def serialize_item_v5(norm, *, max_chars):
        return f"{norm.item_id}|{norm.category}|{norm.name}|{norm.attributes}"[:max_chars]


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(range(1, 42)),
            "name": [f"name-{i}" for i in range(1, 42)],
            "attributes": [{"k": [str(i), "x"]} for i in range(1, 42)],
            "category": ["a" if i % 2 else "b" for i in range(1, 42)],
        }
    )


def test_dual_cache_matches_two_legacy_serial_passes_exactly():
    from ecup_matching.submission.predict_v5 import _legacy_text_cache
    from ecup_matching.submission.v6_text_cache import build_dual_text_cache

    items = _items()
    shared = {}
    expected_contrastive = _legacy_text_cache(
        items, _TextNorm, _ItemText, teacher=False, norm_cache=shared
    )
    expected_teacher = _legacy_text_cache(
        items, _TextNorm, _ItemText, teacher=True, norm_cache=shared
    )

    actual_contrastive, actual_teacher = build_dual_text_cache(
        items,
        _TextNorm,
        _ItemText,
        workers=1,
    )
    assert actual_contrastive == expected_contrastive
    assert actual_teacher == expected_teacher


def test_parallel_dual_cache_preserves_every_item_and_exact_text():
    from ecup_matching.submission.v6_text_cache import build_dual_text_cache

    items = _items()
    serial = build_dual_text_cache(items, _TextNorm, _ItemText, workers=1)
    parallel = build_dual_text_cache(items, _TextNorm, _ItemText, workers=2, chunk_size=7)
    assert parallel == serial
    assert set(parallel[0]) == set(items["id"])
    assert set(parallel[1]) == set(items["id"])


def test_predict_v6_uses_prebuilt_dual_cache_instead_of_two_legacy_passes():
    from pathlib import Path
    source = Path("ecup_matching/submission/predict_v6.py").read_text(encoding="utf-8")
    assert "build_dual_text_cache" in source
    assert "contrastive_text_cache" in source
    assert "teacher_text_cache" in source
