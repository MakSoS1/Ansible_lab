from __future__ import annotations

import math

import numpy as np
import pandas as pd


def select_disagreement_hard_examples(
    frame: pd.DataFrame,
    predictions,
    *,
    max_rows: int,
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"id1", "id2", "target", "weak_weight", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"hard-mining frame missing columns: {sorted(missing)}")
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    work = frame.reset_index(drop=True).copy()
    pred = np.asarray(predictions, dtype=float)
    if pred.shape != (len(work),):
        raise ValueError(f"predictions shape {pred.shape} != ({len(work)},)")
    if not np.isfinite(pred).all():
        raise ValueError("predictions must be finite")
    target = pd.to_numeric(work["target"], errors="raise").to_numpy(float)
    weight = pd.to_numeric(work["weak_weight"], errors="raise").to_numpy(float)
    if "hard_target" not in work.columns:
        work["hard_target"] = (target >= 0.5).astype(np.int8)
    work["_hard_disagreement"] = np.abs(pred - target) * np.clip(weight, 0.0, 1.0)
    work["_hard_prediction"] = pred
    work["_stable_row"] = np.arange(len(work), dtype=np.int64)

    take_total = min(int(max_rows), len(work))
    groups = list(work.groupby(["category", "hard_target"], sort=True, dropna=False))
    quota = max(1, take_total // max(1, len(groups)))
    selected_indices: list[int] = []
    selected_set: set[int] = set()

    # Tiny deterministic jitter breaks exact score ties without making the
    # underlying ranking seed-dependent in ordinary cases.
    rng = np.random.default_rng(int(seed))
    work["_tie"] = rng.random(len(work)) * 1e-12

    for _, group in groups:
        ordered = group.sort_values(
            ["_hard_disagreement", "_tie", "_stable_row"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        for idx in ordered.index[: min(quota, len(ordered))]:
            i = int(idx)
            if i not in selected_set and len(selected_indices) < take_total:
                selected_indices.append(i)
                selected_set.add(i)

    if len(selected_indices) < take_total:
        remaining = work.loc[~work.index.isin(selected_set)].sort_values(
            ["_hard_disagreement", "_tie", "_stable_row"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        need = take_total - len(selected_indices)
        selected_indices.extend(int(i) for i in remaining.index[:need])

    selected = work.loc[selected_indices].copy().reset_index(drop=True)
    selected = selected.drop(columns=["_stable_row", "_tie"])
    disagreement = selected["_hard_disagreement"].to_numpy(float)
    report: dict[str, object] = {
        "input_rows": int(len(work)),
        "selected_rows": int(len(selected)),
        "group_count": int(len(groups)),
        "mean_disagreement": float(disagreement.mean()) if len(disagreement) else 0.0,
        "max_disagreement": float(disagreement.max()) if len(disagreement) else 0.0,
        "seed": int(seed),
    }
    if not math.isfinite(float(report["mean_disagreement"])):
        raise RuntimeError("non-finite hard-mining report")
    return selected, report


__all__ = ["select_disagreement_hard_examples"]
