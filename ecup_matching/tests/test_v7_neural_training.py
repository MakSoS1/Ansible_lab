import numpy as np
import pandas as pd

from ecup_matching.ml.v7_neural import (
    MacroPairBatchSampler,
    build_v7_text_cache_from_parquet,
    phase_microbatches,
)


def _frame():
    return pd.DataFrame(
        {
            "category": ["a"] * 8 + ["b"] * 2,
            "target": [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        }
    )


def test_macro_sampler_equalizes_category_batch_counts_and_mixes_classes():
    frame = _frame()
    sampler = MacroPairBatchSampler(frame, batch_size=2, seed=2026)
    batches = list(iter(sampler))
    assert len(batches) == len(sampler)
    cats = []
    for batch in batches:
        assert len(batch) == 2
        rows = frame.iloc[batch]
        assert rows["category"].nunique() == 1
        cats.append(rows["category"].iloc[0])
        assert set(rows["target"].astype(int)) == {0, 1}
    values, counts = np.unique(cats, return_counts=True)
    assert set(values) == {"a", "b"}
    assert len(set(counts.tolist())) == 1


def test_macro_sampler_is_epoch_deterministic_but_changes_order_between_epochs():
    frame = _frame()
    a = list(MacroPairBatchSampler(frame, batch_size=2, seed=7, epoch=0))
    b = list(MacroPairBatchSampler(frame, batch_size=2, seed=7, epoch=0))
    c = list(MacroPairBatchSampler(frame, batch_size=2, seed=7, epoch=1))
    assert a == b
    assert a != c


def test_phase_microbatches_supports_fractional_epochs_without_zero_work():
    assert phase_microbatches(loader_batches=100, epochs=0.25) == 25
    assert phase_microbatches(loader_batches=3, epochs=0.01) == 1


def test_weak_parquet_text_cache_retains_canonical_attributes(tmp_path):
    items = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": [
                "Samsung Galaxy S24 SM-S921B",
                "Samsung Galaxy S24 SM-S921B",
                "irrelevant",
            ],
            "attributes": [
                '{"Бренд":"Samsung","Модель":"SM-S921B","Встроенная память":"128 GB","Емкость аккумулятора":"4000 mAh"}',
                '{"Бренд":"Samsung","Модель":"SM_S921B","Встроенная память":"0.128 TB","Емкость аккумулятора":"4000 mAh"}',
                '{}',
            ],
            "category": ["phones", "phones", "other"],
        }
    )
    path = tmp_path / "items.parquet"
    items.to_parquet(path, index=False)

    texts, categories = build_v7_text_cache_from_parquet(
        path,
        {1, 2},
        max_chars=360,
        batch_size=2,
    )

    assert set(texts) == {1, 2}
    assert categories == {1: "phones", 2: "phones"}
    assert "storage_bytes_128000000000" in texts[1]
    assert "storage_bytes_128000000000" in texts[2]
    assert "battery_mah_4000" in texts[1]
    assert "[MODEL] sms921b" in texts[1]
    assert "irrelevant" not in " ".join(texts.values())