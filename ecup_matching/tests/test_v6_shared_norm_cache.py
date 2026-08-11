"""The contrastive and teacher text caches must share one ItemNorm pass."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ecup_matching.ml import textnorm as ml_textnorm
from ecup_matching.ml import v5_item_text as ml_item_text
from ecup_matching.submission.predict_v5 import _legacy_text_cache


def _items(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.arange(count, dtype=np.int64),
            "name": [f"смартфон бренд-{i} 128 гб черный" for i in range(count)],
            "attributes": [
                json.dumps({"бренд": [f"b{i % 5}"], "вес": [f"{i} г"]}, ensure_ascii=False)
                for i in range(count)
            ],
            "category": [f"cat_{i % 3}" for i in range(count)],
        }
    )


class _CountingTextnorm:
    def __init__(self):
        self.calls = 0

    def normalize_item(self, item_id, name, attributes, category):
        self.calls += 1
        return ml_textnorm.normalize_item(item_id, name, attributes, category)


def test_shared_norm_cache_produces_identical_texts():
    items = _items(25)
    plain_contrastive = _legacy_text_cache(
        items, ml_textnorm, ml_item_text, teacher=False
    )
    plain_teacher = _legacy_text_cache(items, ml_textnorm, ml_item_text, teacher=True)

    shared: dict[object, object] = {}
    cached_contrastive = _legacy_text_cache(
        items, ml_textnorm, ml_item_text, teacher=False, norm_cache=shared
    )
    cached_teacher = _legacy_text_cache(
        items, ml_textnorm, ml_item_text, teacher=True, norm_cache=shared
    )

    assert cached_contrastive == plain_contrastive
    assert cached_teacher == plain_teacher
    assert cached_contrastive != cached_teacher, "teacher text must keep its category prefix"


def test_shared_norm_cache_normalizes_each_item_once():
    items = _items(25)
    counting = _CountingTextnorm()
    shared: dict[object, object] = {}

    _legacy_text_cache(items, counting, ml_item_text, teacher=False, norm_cache=shared)
    after_first = counting.calls
    _legacy_text_cache(items, counting, ml_item_text, teacher=True, norm_cache=shared)

    assert after_first == len(items)
    assert counting.calls == len(items), (
        "the second pass must reuse cached ItemNorm objects, not renormalize"
    )


def test_missing_cache_keeps_the_original_behaviour():
    items = _items(10)
    counting = _CountingTextnorm()
    _legacy_text_cache(items, counting, ml_item_text, teacher=False)
    _legacy_text_cache(items, counting, ml_item_text, teacher=True)
    assert counting.calls == 2 * len(items)
