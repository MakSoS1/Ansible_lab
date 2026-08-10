from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


ITEM_COLUMNS = ("id", "name", "attributes", "category")


def select_items_by_ids(
    parquet_path: Path,
    item_ids: Iterable[object],
    batch_size: int = 100_000,
) -> pd.DataFrame:
    """Scan a large item parquet and materialize only requested item IDs."""
    requested = set(item_ids)
    if not requested:
        return pd.DataFrame(columns=ITEM_COLUMNS)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    parquet = pq.ParquetFile(str(parquet_path))
    missing_columns = set(ITEM_COLUMNS) - set(parquet.schema_arrow.names)
    if missing_columns:
        raise ValueError(f"items parquet missing columns: {sorted(missing_columns)}")

    pieces: list[pd.DataFrame] = []
    found: set[object] = set()
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(ITEM_COLUMNS)):
        frame = batch.to_pandas()
        selected = frame[frame["id"].isin(requested)]
        if len(selected):
            pieces.append(selected)
            found.update(selected["id"].tolist())
        if found == requested:
            break

    missing = requested - found
    if missing:
        first = min(missing, key=lambda value: (type(value).__name__, repr(value)))
        raise KeyError(f"items parquet is missing {len(missing)} requested IDs; first={first!r}")

    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=ITEM_COLUMNS)
    out = out.drop_duplicates("id", keep="first")
    return out.loc[:, ITEM_COLUMNS].reset_index(drop=True)
