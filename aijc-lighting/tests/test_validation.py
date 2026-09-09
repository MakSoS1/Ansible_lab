import numpy as np

from src.validation import make_splits


def test_stratified_splits_cover_each_validation_row_once():
    y = np.array([0, 1, 2] * 20)
    splits = list(make_splits(y, n_splits=5, seed=42))
    seen = np.zeros(len(y), dtype=int)
    for tr, va in splits:
        assert not set(tr).intersection(set(va))
        seen[va] += 1
        for cls in range(3):
            assert abs((y[va] == cls).mean() - 1 / 3) < 0.15
    assert np.all(seen == 1)


def test_group_splits_never_cross_groups():
    y = np.array([0, 0, 1, 1, 2, 2] * 10)
    groups = np.repeat(np.arange(30), 2)
    splits = list(make_splits(y, groups=groups, n_splits=3, seed=42))
    for tr, va in splits:
        assert set(groups[tr]).isdisjoint(set(groups[va]))


def test_splits_are_reproducible():
    y = np.array([0, 1, 2] * 20)
    a = [(tr.tolist(), va.tolist()) for tr, va in make_splits(y, n_splits=5, seed=42)]
    b = [(tr.tolist(), va.tolist()) for tr, va in make_splits(y, n_splits=5, seed=42)]
    assert a == b
