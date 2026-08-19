import numpy as np
import pandas as pd

from ecup_matching.ml.v2_split import calibration_split, fixed_v1_split


def _matches():
    rows = []
    # 20 disconnected item components; each component has two pair rows.
    for c in range(20):
        base = c * 10
        rows.append((base + 1, base + 2, c % 2, "cat" + str(c % 4)))
        rows.append((base + 2, base + 3, (c + 1) % 2, "cat" + str(c % 4)))
    return pd.DataFrame(rows, columns=["id1", "id2", "target", "category"])


def _items_for(df, idx):
    part = df.iloc[np.asarray(idx)]
    return set(part["id1"]) | set(part["id2"])


def test_fixed_v1_split_is_deterministic_and_item_disjoint():
    df = _matches()
    tr1, va1 = fixed_v1_split(df)
    tr2, va2 = fixed_v1_split(df)
    assert np.array_equal(tr1, tr2)
    assert np.array_equal(va1, va2)
    assert not (_items_for(df, tr1) & _items_for(df, va1))


def test_calibration_is_inside_outer_train_and_disjoint():
    df = _matches()
    outer_train, outer_valid = fixed_v1_split(df)
    fit_idx, calib_idx = calibration_split(df, outer_train, calibration_fraction=0.125)
    assert set(fit_idx).issubset(set(outer_train))
    assert set(calib_idx).issubset(set(outer_train))
    assert not set(fit_idx) & set(calib_idx)
    assert not (_items_for(df, fit_idx) & _items_for(df, calib_idx))
    assert not (_items_for(df, fit_idx) & _items_for(df, outer_valid))
    assert not (_items_for(df, calib_idx) & _items_for(df, outer_valid))
