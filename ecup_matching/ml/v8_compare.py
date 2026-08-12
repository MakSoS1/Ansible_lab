from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .v8_graph import graph_features, graph_rescore
from .v8_selection import anchor_disjoint_split, rank_blend
from .v8_testlike import pseudo_macro_ap_report, soft_rank_report


def _validated_scores(frame: pd.DataFrame, scores: Mapping[str, object]) -> dict[str, np.ndarray]:
    if not scores:
        raise ValueError("at least one score source is required")
    out: dict[str, np.ndarray] = {}
    for name, values in scores.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1 or len(arr) != len(frame):
            raise ValueError(f"score {name!r} must be one-dimensional and aligned")
        if not np.isfinite(arr).all():
            raise ValueError(f"score {name!r} must be finite")
        out[str(name)] = arr
    return out


def _part(frame: pd.DataFrame, mask: np.ndarray, scores: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    pos = np.flatnonzero(mask)
    return frame.iloc[pos].reset_index(drop=True), np.asarray(scores, dtype=np.float64)[pos]


def _diagnostics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, object]:
    return {
        "pseudo": pseudo_macro_ap_report(frame, score),
        "soft": soft_rank_report(frame, score),
    }


def evaluate_grouped_candidates(
    frame: pd.DataFrame,
    scores: Mapping[str, object],
) -> dict[str, object]:
    required = {"id1", "id2", "category", "target"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"grouped evaluator frame missing columns: {sorted(missing)}")
    checked = _validated_scores(frame, scores)
    models = {name: _diagnostics(frame, score) for name, score in checked.items()}
    return {
        "diagnostic_only": True,
        "rows": int(len(frame)),
        "anchors": int(frame["id1"].nunique()),
        "categories": int(frame["category"].astype(str).nunique()),
        "models": models,
        "true_test_ap_claimed": False,
    }


def tune_two_model_blend(
    frame: pd.DataFrame,
    scores: Mapping[str, object],
    *,
    seed: int = 2026,
    weights: Iterable[float] = tuple(np.linspace(0.0, 1.0, 11)),
) -> dict[str, object]:
    checked = _validated_scores(frame, scores)
    if len(checked) != 2:
        raise ValueError("two-model blend requires exactly two score sources")
    names = sorted(checked)
    tune_mask, confirm_mask, split = anchor_disjoint_split(frame, seed=seed, tune_fraction=0.5)
    tune_frame = frame.loc[tune_mask].reset_index(drop=True)
    confirm_frame = frame.loc[confirm_mask].reset_index(drop=True)
    tune_scores = {name: checked[name][tune_mask] for name in names}
    confirm_scores = {name: checked[name][confirm_mask] for name in names}

    candidates: list[dict[str, object]] = []
    seen: set[float] = set()
    for raw in weights:
        first = float(raw)
        if not np.isfinite(first) or first < 0.0 or first > 1.0:
            raise ValueError("blend grid weights must lie in [0,1]")
        if first in seen:
            continue
        seen.add(first)
        mapping = {names[0]: first, names[1]: 1.0 - first}
        blended = rank_blend(tune_frame, tune_scores, mapping)
        pseudo = pseudo_macro_ap_report(tune_frame, blended)
        soft = soft_rank_report(tune_frame, blended)
        candidates.append(
            {
                "weights": mapping,
                "tune_pseudo_ap": float(pseudo["macro_pseudo_average_precision"]),
                "tune_spearman": float(soft["macro_spearman"]),
            }
        )
    if not candidates:
        raise ValueError("blend grid is empty")
    candidates.sort(
        key=lambda row: (
            -float(row["tune_pseudo_ap"]),
            -float(row["tune_spearman"]),
            tuple(float(row["weights"][name]) for name in names),
        )
    )
    selected = candidates[0]
    selected_weights = {name: float(selected["weights"][name]) for name in names}
    confirm_blend = rank_blend(confirm_frame, confirm_scores, selected_weights)
    return {
        "diagnostic_only": True,
        "split": split,
        "selected_weights": selected_weights,
        "tune": {
            "macro_pseudo_average_precision": float(selected["tune_pseudo_ap"]),
            "macro_spearman": float(selected["tune_spearman"]),
        },
        "confirm": _diagnostics(confirm_frame, confirm_blend)["pseudo"]
        | {"macro_spearman": float(soft_rank_report(confirm_frame, confirm_blend)["macro_spearman"])},
        "grid": candidates,
        "true_test_ap_claimed": False,
    }


def _graph_score(score: np.ndarray, features: pd.DataFrame, cfg: Mapping[str, float]) -> np.ndarray:
    return graph_rescore(
        score,
        features,
        reciprocal_best_bonus=float(cfg["rb"]),
        reciprocal_top3_bonus=float(cfg["rt"]),
        endpoint_rank_weight=float(cfg["ep"]),
        ambiguity_penalty=float(cfg["ap"]),
    )


def tune_graph_config(
    frame: pd.DataFrame,
    scores,
    *,
    seed: int = 2026,
    configs: Iterable[Mapping[str, float]] | None = None,
) -> dict[str, object]:
    checked = _validated_scores(frame, {"base": scores})["base"]
    tune_mask, confirm_mask, split = anchor_disjoint_split(frame, seed=seed, tune_fraction=0.5)
    tune_frame, tune_score = _part(frame, tune_mask, checked)
    confirm_frame, confirm_score = _part(frame, confirm_mask, checked)

    # Critical leakage guard: graph features are recomputed independently in
    # tune and confirm, so shared secondary endpoints cannot carry context
    # across the selection boundary.
    tune_features = graph_features(tune_frame[["id1", "id2", "category"]], tune_score)
    confirm_features = graph_features(confirm_frame[["id1", "id2", "category"]], confirm_score)

    if configs is None:
        configs = (
            {"rb": rb, "rt": rt, "ep": ep, "ap": ap}
            for rb in (0.0, 0.005, 0.01, 0.02, 0.03)
            for rt in (0.0, 0.005, 0.01)
            for ep in (0.0, 0.005, 0.01, 0.02, 0.03)
            for ap in (0.0, 0.002, 0.005, 0.01, 0.02)
        )
    rows: list[dict[str, object]] = []
    required = {"rb", "rt", "ep", "ap"}
    for raw in configs:
        if set(raw) != required:
            raise ValueError(f"graph config keys must be exactly {sorted(required)}")
        cfg = {key: float(raw[key]) for key in sorted(required)}
        if not all(np.isfinite(value) for value in cfg.values()):
            raise ValueError("graph config contains nonfinite value")
        rescored = _graph_score(tune_score, tune_features, cfg)
        pseudo = pseudo_macro_ap_report(tune_frame, rescored)
        soft = soft_rank_report(tune_frame, rescored)
        rows.append(
            {
                "config": cfg,
                "tune_pseudo_ap": float(pseudo["macro_pseudo_average_precision"]),
                "tune_spearman": float(soft["macro_spearman"]),
            }
        )
    if not rows:
        raise ValueError("graph config grid is empty")
    rows.sort(
        key=lambda row: (
            -float(row["tune_pseudo_ap"]),
            -float(row["tune_spearman"]),
            tuple(float(row["config"][key]) for key in ("rb", "rt", "ep", "ap")),
        )
    )
    best = rows[0]
    selected = {key: float(best["config"][key]) for key in ("rb", "rt", "ep", "ap")}
    confirm_rescored = _graph_score(confirm_score, confirm_features, selected)
    confirm_diag = _diagnostics(confirm_frame, confirm_rescored)
    base_confirm = _diagnostics(confirm_frame, confirm_score)
    return {
        "diagnostic_only": True,
        "split": split,
        "selected_config": selected,
        "tune": {
            "macro_pseudo_average_precision": float(best["tune_pseudo_ap"]),
            "macro_spearman": float(best["tune_spearman"]),
        },
        "confirm": confirm_diag,
        "base_confirm": base_confirm,
        "confirm_pseudo_delta": float(
            confirm_diag["pseudo"]["macro_pseudo_average_precision"]
            - base_confirm["pseudo"]["macro_pseudo_average_precision"]
        ),
        "confirm_spearman_delta": float(
            confirm_diag["soft"]["macro_spearman"] - base_confirm["soft"]["macro_spearman"]
        ),
        "grid": rows,
        "true_test_ap_claimed": False,
    }


__all__ = ["evaluate_grouped_candidates", "tune_graph_config", "tune_two_model_blend"]
