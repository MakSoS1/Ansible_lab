from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


COLUMNS = ("id1", "id2", "target")


def _weak_weight_vector(target: pd.Series) -> np.ndarray:
    p = pd.to_numeric(target, errors="raise").to_numpy(dtype=float)
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("weak target must be in [0,1]")
    weight = np.zeros(len(p), dtype=np.float32)
    extreme = (p <= 0.03) | (p >= 0.97)
    strong = ((p > 0.03) & (p <= 0.15)) | ((p >= 0.85) & (p < 0.97))
    medium = ((p > 0.15) & (p <= 0.30)) | ((p >= 0.70) & (p < 0.85))
    weight[extreme] = 1.0
    weight[strong] = 0.6
    weight[medium] = 0.3
    return weight


def _eligible_batch(batch, validation_item_ids: set[object]) -> pd.DataFrame:
    out = batch.to_pandas()[list(COLUMNS)].copy()
    out["weak_weight"] = _weak_weight_vector(out["target"])
    keep = out["weak_weight"].to_numpy() > 0
    if validation_item_ids:
        keep &= ~out["id1"].isin(validation_item_ids).to_numpy()
        keep &= ~out["id2"].isin(validation_item_ids).to_numpy()
    return out.loc[keep].reset_index(drop=True)


def prefilter_weak_candidates_parquet(
    path: Path,
    validation_item_ids: set[object],
    max_presample_rows: int,
    seed: int = 2026,
    batch_size: int = 100_000,
) -> tuple[pd.DataFrame, int]:
    """Match pandas sampling while never materializing the full weak parquet.

    The function performs two bounded-memory passes. The first counts eligible
    rows. The second either collects all eligible rows or collects exactly the
    ordinals selected by the same ``RandomState.choice`` operation used by
    ``DataFrame.sample(random_state=seed)``. This preserves deterministic v2b
    semantics while avoiding a >11M-row pandas allocation.
    """
    path = Path(path)
    if max_presample_rows <= 0:
        raise ValueError("max_presample_rows must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    missing = set(COLUMNS) - available
    if missing:
        raise ValueError(f"weak pairs missing columns: {sorted(missing)}")

    input_rows = 0
    eligible_rows = 0
    for batch in parquet.iter_batches(columns=list(COLUMNS), batch_size=batch_size):
        input_rows += int(batch.num_rows)
        eligible_rows += len(_eligible_batch(batch, validation_item_ids))

    if eligible_rows == 0:
        empty = pd.DataFrame(
            {
                "id1": pd.Series(dtype="int64"),
                "id2": pd.Series(dtype="int64"),
                "target": pd.Series(dtype="float64"),
                "weak_weight": pd.Series(dtype="float32"),
                "hard_target": pd.Series(dtype="int8"),
            }
        )
        return empty, input_rows

    choice: np.ndarray | None = None
    if eligible_rows > max_presample_rows:
        random_state = np.random.RandomState(seed)
        choice = random_state.choice(
            eligible_rows,
            size=max_presample_rows,
            replace=False,
        ).astype(np.int64, copy=False)

    pieces: list[pd.DataFrame] = []
    eligible_base = 0
    for batch in parquet.iter_batches(columns=list(COLUMNS), batch_size=batch_size):
        filtered = _eligible_batch(batch, validation_item_ids)
        count = len(filtered)
        if count == 0:
            continue
        if choice is None:
            pieces.append(filtered)
        else:
            in_chunk = (choice >= eligible_base) & (choice < eligible_base + count)
            if in_chunk.any():
                output_positions = np.flatnonzero(in_chunk)
                local_positions = choice[in_chunk] - eligible_base
                selected = filtered.iloc[local_positions].copy()
                selected["_sample_order"] = output_positions
                pieces.append(selected)
        eligible_base += count

    if not pieces:
        raise RuntimeError("streaming weak-label sampling selected no rows")
    out = pd.concat(pieces, ignore_index=True)
    if choice is not None:
        out = (
            out.sort_values("_sample_order", kind="mergesort")
            .drop(columns="_sample_order")
            .reset_index(drop=True)
        )
        if len(out) != max_presample_rows:
            raise RuntimeError(
                f"expected {max_presample_rows} sampled weak rows, got {len(out)}"
            )
    else:
        out = out.reset_index(drop=True)
        if len(out) != eligible_rows:
            raise RuntimeError(
                f"expected {eligible_rows} eligible weak rows, got {len(out)}"
            )
    out["hard_target"] = (out["target"].astype(float) >= 0.5).astype(np.int8)
    return out, input_rows
