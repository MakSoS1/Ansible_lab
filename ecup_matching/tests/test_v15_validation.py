import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from ecup_matching.v15_validate import compute_macro_ap, validate_oof_integrity


def test_v15_macro_ap_is_unweighted_mean_of_category_average_precision():
    frame = pd.DataFrame({
        "row_index": [0, 1, 2, 3, 4, 5],
        "category": ["a", "a", "a", "b", "b", "b"],
        "target": [1, 0, 1, 0, 1, 0],
        "predict": [0.9, 0.8, 0.7, 0.1, 0.2, 0.3],
    })
    expected = np.mean([
        average_precision_score(frame.loc[frame.category == c, "target"], frame.loc[frame.category == c, "predict"])
        for c in ["a", "b"]
    ])
    result = compute_macro_ap(frame, expected_categories=["a", "b"])
    assert result.macro_ap == pytest.approx(expected)
    assert set(result.per_category) == {"a", "b"}


def test_v15_oof_integrity_rejects_duplicate_or_missing_indexes():
    good = pd.DataFrame({"row_index": [0, 1, 2], "predict": [0.1, 0.2, 0.3]})
    validate_oof_integrity(good, expected_row_indexes={0, 1, 2})
    with pytest.raises(ValueError, match="duplicate"):
        validate_oof_integrity(pd.DataFrame({"row_index": [0, 0, 2], "predict": [0.1, 0.2, 0.3]}), expected_row_indexes={0, 1, 2})
    with pytest.raises(ValueError, match="coverage"):
        validate_oof_integrity(good.iloc[:2], expected_row_indexes={0, 1, 2})
