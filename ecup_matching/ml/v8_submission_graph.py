from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .v8_graph import graph_features, graph_rescore


def apply_graph_to_prediction(
    test: pd.DataFrame,
    test_items: pd.DataFrame,
    prediction: pd.DataFrame,
    config: Mapping[str, float],
) -> pd.DataFrame:
    required_test = {"id", "id1", "id2"}
    missing_test = required_test - set(test.columns)
    if missing_test:
        raise ValueError(f"test missing columns: {sorted(missing_test)}")
    if not {"id", "category"}.issubset(test_items.columns):
        raise ValueError("test-items must contain id and category")
    if not {"id", "predict"}.issubset(prediction.columns):
        raise ValueError("prediction must contain id and predict")
    if len(test) != len(prediction):
        raise ValueError("prediction row count mismatch")
    if not np.array_equal(test["id"].to_numpy(), prediction["id"].to_numpy()):
        raise ValueError("prediction id order mismatch")
    required_cfg = {"rb", "rt", "ep", "ap"}
    if set(config) != required_cfg:
        raise ValueError(f"graph config keys must be exactly {sorted(required_cfg)}")
    cfg = {key: float(config[key]) for key in required_cfg}
    if not all(np.isfinite(value) for value in cfg.values()):
        raise ValueError("graph config contains nonfinite value")

    item_categories = test_items[["id", "category"]].drop_duplicates("id", keep=False)
    category_map = dict(
        zip(item_categories["id"].tolist(), item_categories["category"].astype(str).tolist())
    )
    left = test["id1"].map(category_map)
    right = test["id2"].map(category_map)
    if left.isna().any() or right.isna().any():
        raise ValueError("test pair has missing category in test-items")
    if not np.array_equal(left.astype(str).to_numpy(), right.astype(str).to_numpy()):
        raise ValueError("test contains cross-category pair")

    base = pd.to_numeric(prediction["predict"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(base).all():
        raise ValueError("prediction contains nonfinite score")
    graph_frame = pd.DataFrame(
        {
            "id1": test["id1"].to_numpy(),
            "id2": test["id2"].to_numpy(),
            "category": left.astype(str).to_numpy(),
        }
    )
    features = graph_features(graph_frame, base)
    rescored = graph_rescore(
        base,
        features,
        reciprocal_best_bonus=cfg["rb"],
        reciprocal_top3_bonus=cfg["rt"],
        endpoint_rank_weight=cfg["ep"],
        ambiguity_penalty=cfg["ap"],
    )
    if not np.isfinite(rescored).all():
        raise RuntimeError("graph postprocess produced nonfinite score")
    out = prediction.copy().reset_index(drop=True)
    out["predict"] = rescored.astype(np.float32)
    return out


__all__ = ["apply_graph_to_prediction"]
