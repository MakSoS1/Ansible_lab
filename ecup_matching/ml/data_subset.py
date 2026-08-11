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
    batch_size: int = 131_072,
    *,
    include_attributes: bool = True,
) -> pd.DataFrame:
    """Scan a large item parquet and materialize only requested item IDs.

    The requested value set is built once and reused for every record batch.
    Rebuilding it per batch made the scan quadratic in ``len(item_ids)`` and
    dominated submission load time on competition-sized item files.
    """
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

    id_type = parquet.schema_arrow.field("id").type
    value_set = pa.array(list(requested), type=id_type)

    pieces: list[pa.RecordBatch] = []
    found: set[object] = set()
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(scan_columns)):
        id_column = batch.column(batch.schema.get_field_index("id"))
        selected_batch = batch.filter(pc.is_in(id_column, value_set=value_set))
        if selected_batch.num_rows:
            pieces.append(selected_batch)
            found.update(
                selected_batch.column(
                    selected_batch.schema.get_field_index("id")
                ).to_pylist()
            )
        del id_column, selected_batch, batch
        if len(found) == len(requested):
            break

    if pieces:
        table = pa.Table.from_batches(pieces)
        out = table.to_pandas()
        del table
    else:
        out = pd.DataFrame(columns=scan_columns)
    del pieces, value_set
    pa.default_memory_pool().release_unused()

    out = out.drop_duplicates("id", keep="first")
    missing = requested - set(out["id"].tolist())
    if missing:
        first = min(missing, key=lambda value: (type(value).__name__, repr(value)))
        raise KeyError(f"items parquet is missing {len(missing)} requested IDs; first={first!r}")

    if not include_attributes:
        out["attributes"] = "{}"
    return out.loc[:, ITEM_COLUMNS].reset_index(drop=True)
