from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ITEM_COLUMNS = ("id", "name", "attributes", "category")


def select_items_by_ids(
    parquet_path: Path,
    item_ids: Iterable[object],
    batch_size: int = 5_000,
    *,
    include_attributes: bool = True,
) -> pd.DataFrame:
    """Scan a large item parquet and materialize only requested item IDs."""
    requested = set(item_ids)
    if not requested:
        return pd.DataFrame(columns=ITEM_COLUMNS)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    parquet = pq.ParquetFile(str(parquet_path))
    scan_columns = ITEM_COLUMNS if include_attributes else ("id", "name", "category")
    missing_columns = set(scan_columns) - set(parquet.schema_arrow.names)
    if missing_columns:
        raise ValueError(f"items parquet missing columns: {sorted(missing_columns)}")

    pieces: list[pd.DataFrame] = []
    found: set[object] = set()
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(scan_columns)):
        id_column = batch.column(batch.schema.get_field_index("id"))
        requested_values = pa.array(list(requested - found), type=id_column.type)
        selected_batch = batch.filter(pc.is_in(id_column, value_set=requested_values))
        if selected_batch.num_rows:
            selected = selected_batch.to_pandas()
            pieces.append(selected)
            found.update(selected["id"].tolist())
            del selected
        complete = found == requested
        del id_column, requested_values, selected_batch, batch
        pa.default_memory_pool().release_unused()
        if complete:
            break

    missing = requested - found
    if missing:
        first = min(missing, key=lambda value: (type(value).__name__, repr(value)))
        raise KeyError(f"items parquet is missing {len(missing)} requested IDs; first={first!r}")

    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=scan_columns)
    out = out.drop_duplicates("id", keep="first")
    if not include_attributes:
        out["attributes"] = "{}"
    return out.loc[:, ITEM_COLUMNS].reset_index(drop=True)
