import numpy as np
import pandas as pd

from ecup_matching.ml.v7_neural import MacroPairBatchSampler, phase_microbatches


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
