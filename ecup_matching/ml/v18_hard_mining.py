from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _append_unique(
    destination: list[int],
    seen: set[int],
    indices,
    *,
    limit: int,
) -> None:
    for raw in indices:
        index = int(raw)
        if index in seen:
            continue
        destination.append(index)
        seen.add(index)
        if len(destination) >= int(limit):
            break


def select_disagreement_hard_examples(
    frame: pd.DataFrame,
    predictions,
    *,
    max_rows: int,
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a deterministic hard+broad weak mixture for the second curriculum phase.

    Roughly half of the budget is filled by high model/weak-label disagreement,
    balanced across category and hard class. The other half is deterministic
    broad coverage from the remaining weak distribution. Targets are never
    replaced by model predictions; predictions are selection evidence only.
    """
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

    # Tiny deterministic jitter breaks exact disagreement ties while retaining
    # a reproducible order for the same seed. It must exist before groupby:
    # pandas materializes the grouped frames, so columns added later are absent
    # from those group frames and cannot be used by sort_values.
    rng = np.random.default_rng(int(seed))
    work["_tie"] = rng.random(len(work)) * 1e-12

    take_total = min(int(max_rows), len(work))
    hard_budget = min(take_total, max(1, int(math.ceil(take_total * 0.5))))
    broad_budget = take_total - hard_budget
    groups = list(work.groupby(["category", "hard_target"], sort=True, dropna=False))
    hard_quota = max(1, int(math.ceil(hard_budget / max(1, len(groups)))))
    selected_indices: list[int] = []
    selected_set: set[int] = set()

    # Part 1: model-disagreement examples, approximately category/class balanced.
    for _, group in groups:
        if len(selected_indices) >= hard_budget:
            break
        ordered = group.sort_values(
            ["_hard_disagreement", "_tie", "_stable_row"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        _append_unique(
            selected_indices,
            selected_set,
            ordered.index[: min(hard_quota, len(ordered))],
            limit=hard_budget,
        )
    if len(selected_indices) < hard_budget:
        remaining_hard = work.loc[~work.index.isin(selected_set)].sort_values(
            ["_hard_disagreement", "_tie", "_stable_row"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        _append_unique(
            selected_indices,
            selected_set,
            remaining_hard.index,
            limit=hard_budget,
        )
    actual_hard_rows = int(len(selected_indices))

    # Part 2: broad coverage from the rows not already selected. Sampling is
    # stratified by category/class so the second phase does not collapse to only
    # pathological disagreements and forget the wider weak distribution.
    if broad_budget > 0:
        remaining = work.loc[~work.index.isin(selected_set)]
        remaining_groups = list(
            remaining.groupby(["category", "hard_target"], sort=True, dropna=False)
        )
        broad_quota = max(1, int(math.ceil(broad_budget / max(1, len(remaining_groups)))))
        broad_limit = hard_budget + broad_budget
        for group_number, (_, group) in enumerate(remaining_groups):
            if len(selected_indices) >= broad_limit:
                break
            take = min(broad_quota, len(group))
            if take <= 0:
                continue
            sampled = group.sample(n=take, random_state=int(seed) + 10_007 + group_number)
            _append_unique(
                selected_indices,
                selected_set,
                sampled.index,
                limit=broad_limit,
            )
        if len(selected_indices) < broad_limit:
            pool = work.loc[~work.index.isin(selected_set)]
            if len(pool):
                need = min(broad_limit - len(selected_indices), len(pool))
                sampled = pool.sample(n=need, random_state=int(seed) + 20_011)
                _append_unique(
                    selected_indices,
                    selected_set,
                    sampled.index,
                    limit=broad_limit,
                )

    selected = work.loc[selected_indices].copy().reset_index(drop=True)
    selected["_selection_role"] = [
        "hard" if index < actual_hard_rows else "broad"
        for index in range(len(selected))
    ]
    selected = selected.drop(columns=["_stable_row", "_tie"])
    disagreement = selected["_hard_disagreement"].to_numpy(float)
    report: dict[str, object] = {
        "input_rows": int(len(work)),
        "selected_rows": int(len(selected)),
        "hard_rows": int((selected["_selection_role"] == "hard").sum()),
        "broad_rows": int((selected["_selection_role"] == "broad").sum()),
        "group_count": int(len(groups)),
        "mean_disagreement": float(disagreement.mean()) if len(disagreement) else 0.0,
        "max_disagreement": float(disagreement.max()) if len(disagreement) else 0.0,
        "seed": int(seed),
    }
    if not math.isfinite(float(report["mean_disagreement"])):
        raise RuntimeError("non-finite hard-mining report")
    return selected, report


__all__ = ["select_disagreement_hard_examples"]
