"""Inference-only half of the v7 cross-encoder.

Split out of ``v7_neural`` so the submission archive does not have to ship the
training graph. ``v7_neural`` imports ``train_v5_teacher_fold`` and
``v5_teacher2_objective``, which transitively pull in the split, metric and
validation modules; packaging those into an offline archive is what previously
broke the organizer smoke with a training-only import.

Both training and inference import these functions from here, so the serialized
text and the scoring loop cannot drift apart between the two.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .textnorm import normalize_item
from .v7_item_text import serialize_item_v7


def item_text_v7(
    item_id: object,
    name: object,
    attributes: object,
    category: object,
    *,
    max_chars: int,
    attribute_importance: Mapping[str, float] | None = None,
) -> tuple[str, str]:
    """Serialize one item exactly as training does. Returns (text, category)."""
    norm = normalize_item(item_id, name, attributes, category)
    text = (
        f"[CAT] {norm.category}\n"
        f"{serialize_item_v7(norm, max_chars=max_chars, attribute_importance=attribute_importance)}"
    )
    return text, str(category)


def build_v7_text_cache_from_parquet(
    parquet_path: Path,
    item_ids: Iterable[object],
    *,
    max_chars: int,
    batch_size: int = 131_072,
    attribute_importance: Mapping[str, float] | None = None,
) -> tuple[dict[object, str], dict[object, str]]:
    """Stream selected full item records and serialize them without a giant DataFrame.

    v7 depends on canonical typed attributes. The legacy
    ``include_attributes=False`` fast path intentionally replaced attributes by
    ``{}``; that is correct for name-only consumers but silently defeats the v7
    hypothesis. This scanner keeps the real attributes while retaining only the
    final serialized strings and category map in memory.
    """
    requested = set(item_ids)
    if not requested:
        return {}, {}
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    parquet = pq.ParquetFile(str(parquet_path))
    columns = ("id", "name", "attributes", "category")
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"items parquet missing columns: {sorted(missing)}")
    id_type = parquet.schema_arrow.field("id").type
    value_set = pa.array(list(requested), type=id_type)
    texts: dict[object, str] = {}
    categories: dict[object, str] = {}
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
        id_column = batch.column(batch.schema.get_field_index("id"))
        selected = batch.filter(pc.is_in(id_column, value_set=value_set))
        if selected.num_rows:
            payload = selected.to_pydict()
            for item_id, name, attributes, category in zip(
                payload["id"],
                payload["name"],
                payload["attributes"],
                payload["category"],
            ):
                if item_id in texts:
                    continue
                texts[item_id], categories[item_id] = item_text_v7(
                    item_id,
                    name,
                    attributes,
                    category,
                    max_chars=max_chars,
                    attribute_importance=attribute_importance,
                )
        del id_column, selected, batch
        if len(texts) == len(requested):
            break
    missing_ids = requested - set(texts)
    if missing_ids:
        first = min(missing_ids, key=lambda value: (type(value).__name__, repr(value)))
        raise KeyError(
            f"items parquet is missing {len(missing_ids)} requested IDs; first={first!r}"
        )
    del value_set
    pa.default_memory_pool().release_unused()
    return texts, categories


def predict_pairs(
    *,
    model,
    tokenizer,
    frame: pd.DataFrame,
    texts: Mapping[object, str],
    device: str,
    max_length: int,
    batch_size: int = 16,
) -> tuple[np.ndarray, dict[str, float | int]]:
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    predictions: list[np.ndarray] = []
    started = time.perf_counter()
    current_batch = int(batch_size)
    row = 0
    on_cuda = device.startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        while row < len(frame):
            stop = min(len(frame), row + current_batch)
            chunk = frame.iloc[row:stop]
            try:
                tokens = tokenizer(
                    [texts[item_id] for item_id in chunk["id1"]],
                    [texts[item_id] for item_id in chunk["id2"]],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokens = {
                    key: value.to(device, non_blocking=on_cuda)
                    for key, value in tokens.items()
                }
                with torch.autocast(
                    device_type="cuda" if on_cuda else "cpu",
                    dtype=torch.float16,
                    enabled=on_cuda,
                ):
                    logits = model(**tokens).logits.squeeze(-1)
                predictions.append(torch.sigmoid(logits).float().cpu().numpy())
                row = stop
            except torch.cuda.OutOfMemoryError:
                if not on_cuda or current_batch <= 1:
                    raise
                current_batch = max(1, current_batch // 2)
                torch.cuda.empty_cache()
                print(
                    {"phase": "predict", "cuda_oom_batch_halved_to": current_batch},
                    flush=True,
                )
    if on_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    score = (
        np.concatenate(predictions).astype(np.float64, copy=False)
        if predictions
        else np.empty(0, dtype=np.float64)
    )
    return score, {
        "rows": int(len(frame)),
        "seconds": float(elapsed),
        "examples_per_second": float(len(frame) / max(elapsed, 1e-9)),
        "batch_size_final": int(current_batch),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if on_cuda else 0,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if on_cuda else 0,
    }
