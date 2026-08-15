"""Fold-safe, legacy-target-free distillation contracts for E-CUP v15."""

from __future__ import annotations

from collections.abc import Iterable
import pandas as pd


def select_unlabelled_candidates(reader, *, excluded_item_ids: set[int] | Iterable[int], limit: int) -> pd.DataFrame:
    """Read only endpoint topology and return a deterministic safe subset.

    ``reader`` intentionally exposes a tiny ``read(columns=...)`` contract so
    tests can prove the legacy LLM ``target`` column is never requested.
    """

    if limit < 0:
        raise ValueError("limit must be non-negative")
    frame = reader.read(columns=["id1", "id2"])
    if list(frame.columns) != ["id1", "id2"]:
        frame = frame.loc[:, ["id1", "id2"]]

    protected = set(excluded_item_ids)
    if protected:
        mask = ~frame["id1"].isin(protected) & ~frame["id2"].isin(protected)
        frame = frame.loc[mask]
    frame = frame.loc[frame["id1"] != frame["id2"]]
    # Preserve retrieval source order for reproducibility; do not canonicalize
    # endpoints and therefore do not destroy original anchor orientation.
    return frame.head(limit).reset_index(drop=True)


def rank_active_candidates(frame: pd.DataFrame, *, score_columns: Iterable[str]) -> pd.DataFrame:
    """Deterministically prioritize informative candidates by supplied scores.

    The caller computes teacher/student/typed disagreement columns. This helper
    never accesses labels and uses a stable source-order tie break.
    """

    columns = list(score_columns)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"missing active-sampling columns: {missing}")
    work = frame.copy()
    work["_source_order"] = range(len(work))
    work["_active_score"] = work[columns].fillna(0.0).astype(float).sum(axis=1)
    work = work.sort_values(["_active_score", "_source_order"], ascending=[False, True], kind="mergesort")
    return work.drop(columns=["_source_order", "_active_score"]).reset_index(drop=True)
