from __future__ import annotations

import multiprocessing
from typing import Any

import pandas as pd

from .v6_fast import batch_index_ranges
from .v6_parallel import parallel_supported, resolve_worker_count


_REQUIRED_COLUMNS = ("id", "name", "attributes", "category")
_WORKER_ITEMS: pd.DataFrame | None = None
_WORKER_TEXTNORM: Any = None
_WORKER_ITEM_TEXT: Any = None


def _serialize_rows(items: pd.DataFrame, legacy_textnorm, legacy_item_text) -> list[tuple[object, str]]:
    result: list[tuple[object, str]] = []
    rows = items.loc[:, list(_REQUIRED_COLUMNS)]
    for item_id, name, attributes, category in rows.itertuples(index=False, name=None):
        norm = legacy_textnorm.normalize_item(item_id, name, attributes, category)
        text = legacy_item_text.serialize_item_v5(norm, max_chars=700)
        result.append((item_id, text))
    return result


def _worker_entry(bounds: tuple[int, int]) -> list[tuple[object, str]]:
    if _WORKER_ITEMS is None or _WORKER_TEXTNORM is None or _WORKER_ITEM_TEXT is None:
        raise RuntimeError("v10 text-cache worker was not initialized")
    start, end = bounds
    return _serialize_rows(
        _WORKER_ITEMS.iloc[start:end],
        _WORKER_TEXTNORM,
        _WORKER_ITEM_TEXT,
    )


def _merge(payloads) -> dict[object, str]:
    result: dict[object, str] = {}
    for payload in payloads:
        for item_id, text in payload:
            result[item_id] = text
    return result


def build_contrastive_text_cache(
    items: pd.DataFrame,
    legacy_textnorm,
    legacy_item_text,
    *,
    workers: int | None = None,
    chunk_size: int | None = None,
) -> dict[object, str]:
    """Build only the byte-identical v5 contrastive text view.

    v10 has no pair-teacher checkpoint, so generating its 850-character text
    representation is pure waste.  This keeps the exact 700-character
    contrastive serialization while removing the unused second string and its
    memory traffic.
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in items.columns]
    if missing:
        raise ValueError(f"items missing required text-cache columns: {missing}")
    row_count = int(len(items))
    if row_count == 0:
        return {}

    worker_count = resolve_worker_count(workers)
    if chunk_size is None:
        chunk_size = max(1_000, (row_count + worker_count * 4 - 1) // (worker_count * 4))
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    bounds = list(batch_index_ranges(row_count, chunk_size))

    if worker_count <= 1 or len(bounds) <= 1 or not parallel_supported():
        return _merge([_serialize_rows(items, legacy_textnorm, legacy_item_text)])

    global _WORKER_ITEMS, _WORKER_TEXTNORM, _WORKER_ITEM_TEXT
    _WORKER_ITEMS = items
    _WORKER_TEXTNORM = legacy_textnorm
    _WORKER_ITEM_TEXT = legacy_item_text
    try:
        context = multiprocessing.get_context("fork")
        try:
            pool = context.Pool(processes=min(worker_count, len(bounds)))
        except OSError as error:
            print(
                f"[v10-faststack] text-cache pool unavailable ({error}); falling back to serial",
                flush=True,
            )
            return _merge([_serialize_rows(items, legacy_textnorm, legacy_item_text)])
        with pool:
            result = _merge(pool.imap(_worker_entry, bounds, chunksize=1))
    finally:
        _WORKER_ITEMS = None
        _WORKER_TEXTNORM = None
        _WORKER_ITEM_TEXT = None
    return result


__all__ = ["build_contrastive_text_cache"]
