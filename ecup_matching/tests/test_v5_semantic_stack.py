import numpy as np
import pandas as pd

from ecup_matching.ml.v5_semantic_stack import crossfit_semantic_stack


def test_semantic_stack_is_strictly_oof_and_can_correct_a_bad_anchor():
    rows = []
    base = []
    semantic = []
    folds = []
    # Every fold contains both classes in both categories. The base score is
    # intentionally reversed while semantic similarity is perfectly ordered.
    for fold in range(5):
        for category in ("a", "b"):
            for target in (0, 1):
                for rep in range(8):
                    rows.append({"target": target, "category": category})
                    base.append(0.8 if target == 0 else 0.2)
                    semantic.append(
                        [
                            0.95 if target else 0.05,
                            0.02 if target else 0.90,
                        ]
                    )
                    folds.append(fold)
    frame = pd.DataFrame(rows)
    semantic_frame = pd.DataFrame(semantic, columns=["semantic_cosine", "semantic_distance"])

    result = crossfit_semantic_stack(
        frame,
        np.asarray(base, dtype=float),
        semantic_frame,
        np.asarray(folds, dtype=int),
        seed=2026,
    )

    assert result["scores"].shape == (len(frame),)
    assert np.isfinite(result["scores"]).all()
    assert result["macro_average_precision"] > result["base_macro_average_precision"]
    assert len(result["fold_reports"]) == 5
    assert all(report["train_rows"] == len(frame) - 32 for report in result["fold_reports"])
    assert all(report["valid_rows"] == 32 for report in result["fold_reports"])


def test_semantic_stack_rejects_missing_or_single_fold_inputs():
    frame = pd.DataFrame({"target": [0, 1], "category": ["a", "a"]})
    semantic = pd.DataFrame({"semantic_cosine": [0.1, 0.9]})

    for bad_folds in (np.array([0]), np.array([0, 0])):
        try:
            crossfit_semantic_stack(frame, np.array([0.2, 0.8]), semantic, bad_folds)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid fold assignment must be rejected")
