"""Parallel serialization and tokenizer prefetch must not change any output.

Both fixes target CPU work that a faster accelerator does not shrink, so they
sit on the critical path of the submission regardless of the GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ecup_matching.ml.v7_runtime import (
    _BatchPrefetcher,
    build_v7_text_cache_from_parquet,
    serialization_workers,
)


def _items_parquet(path: Path, rows: int, row_group_size: int = 50) -> None:
    brands = ["samsung", "xiaomi", "bosch", "philips"]
    table = pa.table(
        {
            "id": pa.array(np.arange(rows, dtype=np.int64)),
            "name": pa.array(
                [
                    f"смартфон {brands[i % 4]} sm-s{900 + i % 40}b {128 * (1 + i % 3)} гб"
                    for i in range(rows)
                ]
            ),
            "attributes": pa.array(
                [
                    json.dumps(
                        {
                            "Бренд": [brands[i % 4]],
                            "Цвет товара": ["черный" if i % 2 else "белый"],
                            "Объем встроенной памяти": [f"{128 * (1 + i % 3)} ГБ"],
                            "Вес товара, г": [f"{i % 400} г"],
                        },
                        ensure_ascii=False,
                    )
                    for i in range(rows)
                ]
            ),
            "category": pa.array(
                [["Электроника", "Обувь", "Одежда"][i % 3] for i in range(rows)]
            ),
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)


# --- parallel item serialization -------------------------------------------


def test_parallel_serialization_matches_the_serial_scan(tmp_path):
    path = tmp_path / "items.parquet"
    _items_parquet(path, rows=400, row_group_size=40)
    needed = list(range(0, 400, 3))

    serial_texts, serial_cats = build_v7_text_cache_from_parquet(
        path, needed, max_chars=900, workers=1
    )
    parallel_texts, parallel_cats = build_v7_text_cache_from_parquet(
        path, needed, max_chars=900, workers=4
    )

    assert serial_texts == parallel_texts
    assert serial_cats == parallel_cats
    assert len(serial_texts) == len(needed)


def test_parallel_serialization_is_independent_of_worker_count(tmp_path):
    path = tmp_path / "items.parquet"
    _items_parquet(path, rows=300, row_group_size=25)
    needed = list(range(300))

    reference, _ = build_v7_text_cache_from_parquet(path, needed, max_chars=900, workers=1)
    for workers in (2, 3, 8):
        candidate, _ = build_v7_text_cache_from_parquet(
            path, needed, max_chars=900, workers=workers
        )
        assert candidate == reference, f"serialization changed with workers={workers}"


def test_serialization_still_reports_missing_items(tmp_path):
    path = tmp_path / "items.parquet"
    _items_parquet(path, rows=100, row_group_size=20)
    with pytest.raises(KeyError, match="missing"):
        build_v7_text_cache_from_parquet(path, [1, 2, 9999], max_chars=900, workers=4)


def test_serialization_keeps_real_attributes(tmp_path):
    """The v7 hypothesis dies quietly if attributes are dropped."""
    path = tmp_path / "items.parquet"
    _items_parquet(path, rows=60, row_group_size=10)
    texts, _ = build_v7_text_cache_from_parquet(path, [0, 1, 2], max_chars=900, workers=2)
    joined = " ".join(texts.values())
    assert "[IDENTITY]" in joined
    assert "гб" in joined.lower() or "storage" in joined.lower()


def test_serialization_worker_count_is_bounded():
    assert serialization_workers(10, cpu_count=20) == 1, "tiny inputs stay serial"
    assert serialization_workers(1_000_000, cpu_count=1) == 1
    assert serialization_workers(1_000_000, cpu_count=2) == 1
    # Capped at 8: each worker decodes a whole row group including attributes.
    assert serialization_workers(1_000_000, cpu_count=20) == 8
    assert serialization_workers(60_000, cpu_count=20) == 3
    assert serialization_workers(1_000_000, cpu_count=1000) <= 8


def test_empty_request_returns_empty_caches(tmp_path):
    path = tmp_path / "items.parquet"
    _items_parquet(path, rows=10, row_group_size=5)
    assert build_v7_text_cache_from_parquet(path, [], max_chars=900) == ({}, {})


# --- tokenizer prefetch -----------------------------------------------------


def test_prefetcher_returns_every_batch_in_order():
    calls: list[tuple[int, int]] = []

    def tokenize(start, stop):
        calls.append((start, stop))
        return {"input_ids": list(range(start, stop))}

    prefetch = _BatchPrefetcher(tokenize, total_rows=10, batch_size=4)
    try:
        assert prefetch.get(0, 4)["input_ids"] == [0, 1, 2, 3]
        assert prefetch.get(4, 8)["input_ids"] == [4, 5, 6, 7]
        assert prefetch.get(8, 10)["input_ids"] == [8, 9]
    finally:
        prefetch.close()


def test_prefetcher_survives_a_batch_size_change_midstream():
    def tokenize(start, stop):
        return {"input_ids": list(range(start, stop))}

    prefetch = _BatchPrefetcher(tokenize, total_rows=10, batch_size=4)
    assert prefetch.get(0, 4)["input_ids"] == [0, 1, 2, 3]
    prefetch.close()

    # An OOM halves the batch and restarts the producer from the failed row.
    resumed = _BatchPrefetcher(tokenize, total_rows=10, batch_size=2, start=4)
    try:
        assert resumed.get(4, 6)["input_ids"] == [4, 5]
        assert resumed.get(6, 8)["input_ids"] == [6, 7]
    finally:
        resumed.close()


def test_prefetcher_propagates_producer_errors():
    def tokenize(start, stop):
        raise ValueError("tokenizer exploded")

    prefetch = _BatchPrefetcher(tokenize, total_rows=4, batch_size=2)
    try:
        with pytest.raises(ValueError, match="tokenizer exploded"):
            prefetch.get(0, 2)
    finally:
        prefetch.close()


def test_prefetcher_falls_back_when_bounds_do_not_match():
    """A mismatch may cost speed but must never score the wrong rows."""
    def tokenize(start, stop):
        return {"input_ids": list(range(start, stop))}

    prefetch = _BatchPrefetcher(tokenize, total_rows=8, batch_size=4)
    try:
        out = prefetch.get(100, 104)
        assert out["input_ids"] == [100, 101, 102, 103]
    finally:
        prefetch.close()


def test_prefetcher_close_is_idempotent():
    prefetch = _BatchPrefetcher(lambda s, e: {"x": s}, total_rows=4, batch_size=2)
    prefetch.close()
    prefetch.close()


def test_serialization_is_serial_unless_the_caller_opts_in(tmp_path, monkeypatch):
    """Training scans the 13M-row item file; forked workers there caused an OOM kill."""
    path = tmp_path / "items.parquet"
    _items_parquet(path, rows=200, row_group_size=20)
    seen: list[int] = []
    real = __import__(
        "ecup_matching.ml.v7_runtime", fromlist=["serialization_workers"]
    ).serialization_workers
    monkeypatch.setattr(
        "ecup_matching.ml.v7_runtime.serialization_workers",
        lambda *a, **k: seen.append(1) or real(*a, **k),
    )
    build_v7_text_cache_from_parquet(path, list(range(200)), max_chars=900)
    assert not seen, "default path must not consult the worker heuristic at all"


def test_worker_cap_bounds_row_group_memory():
    """Each worker decodes a whole row group with attributes, so the cap is memory."""
    assert serialization_workers(10_000_000, cpu_count=64) <= 8
    assert serialization_workers(10_000_000, cpu_count=64, max_workers=2) == 2


def test_submission_path_opts_in_to_parallel_serialization():
    source = (
        Path(__file__).resolve().parents[1] / "submission" / "predict_v7.py"
    ).read_text(encoding="utf-8")
    assert "workers=workers" in source
    assert "serialization_workers(len(needed))" in source
