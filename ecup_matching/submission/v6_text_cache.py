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


def _serialize_rows(
    items: pd.DataFrame,
    legacy_textnorm,
    legacy_item_text,
) -> list[tuple[object, str, str]]:
    """Normalize once and emit the exact two legacy neural serializations."""
    result: list[tuple[object, str, str]] = []
    rows = items.loc[:, list(_REQUIRED_COLUMNS)]
    for item_id, name, attributes, category in rows.itertuples(index=False, name=None):
        norm = legacy_textnorm.normalize_item(item_id, name, attributes, category)
        contrastive = legacy_item_text.serialize_item_v5(norm, max_chars=700)
        teacher_body = legacy_item_text.serialize_item_v5(norm, max_chars=850)
        teacher = f"[CAT] {norm.category}\n{teacher_body}"
        result.append((item_id, contrastive, teacher))
    return result


def _worker_entry(bounds: tuple[int, int]) -> list[tuple[object, str, str]]:
    if _WORKER_ITEMS is None or _WORKER_TEXTNORM is None or _WORKER_ITEM_TEXT is None:
        raise RuntimeError("dual text-cache worker was not initialized")
    start, end = bounds
    return _serialize_rows(
        _WORKER_ITEMS.iloc[start:end],
        _WORKER_TEXTNORM,
        _WORKER_ITEM_TEXT,
    )


def _merge_payloads(
    payloads,
) -> tuple[dict[object, str], dict[object, str]]:
    contrastive: dict[object, str] = {}
    teacher: dict[object, str] = {}
    for payload in payloads:
        for item_id, contrastive_text, teacher_text in payload:
            # Processing payloads in original chunk order preserves the legacy
            # dict-overwrite semantics even if a malformed input contains a
            # duplicate item id.
            contrastive[item_id] = contrastive_text
            teacher[item_id] = teacher_text
    return contrastive, teacher


def build_dual_text_cache(
    items: pd.DataFrame,
    legacy_textnorm,
    legacy_item_text,
    *,
    workers: int | None = None,
    chunk_size: int | None = None,
) -> tuple[dict[object, str], dict[object, str]]:
    """Build contrastive and teacher text caches in one exact normalization pass.

    The legacy runtime previously normalized every item during the contrastive
    cache pass and then serialized every item again for the teacher cache.  The
    two model inputs differ only by max_chars and the teacher category prefix,
    so one normalization can deterministically emit both strings.  On Linux we
    distribute independent item ranges over fork workers; the parent merges
    results in original range order.  This changes only preprocessing wall
    time: the returned strings are byte-for-byte identical to two calls to
    ``_legacy_text_cache`` sharing a normalization cache.
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in items.columns]
    if missing:
        raise ValueError(f"items missing required text-cache columns: {missing}")
    row_count = int(len(items))
    if row_count == 0:
        return {}, {}

    worker_count = resolve_worker_count(workers)
    if chunk_size is None:
        # More chunks than workers keep the tail balanced without creating
        # thousands of large pickle payloads for a 500k+ item catalogue.
        chunk_size = max(1_000, (row_count + worker_count * 4 - 1) // (worker_count * 4))
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    bounds = list(batch_index_ranges(row_count, chunk_size))

    if worker_count <= 1 or len(bounds) <= 1 or not parallel_supported():
        return _merge_payloads(
            [_serialize_rows(items, legacy_textnorm, legacy_item_text)]
        )

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
                f"[v6] text-cache pool unavailable ({error}); falling back to serial",
                flush=True,
            )
            return _merge_payloads(
                [_serialize_rows(items, legacy_textnorm, legacy_item_text)]
            )
        with pool:
            # imap preserves bounds order, so dict overwrite/order behavior is
            # the same as the original serial traversal.
            result = _merge_payloads(pool.imap(_worker_entry, bounds, chunksize=1))
    finally:
        _WORKER_ITEMS = None
        _WORKER_TEXTNORM = None
        _WORKER_ITEM_TEXT = None
    return result


__all__ = ["build_dual_text_cache"]
