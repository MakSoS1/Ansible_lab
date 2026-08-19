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

import multiprocessing
import os
from pathlib import Path
import queue
import threading
import time
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .progress import ProgressReporter
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


def serialization_workers(
    requested_items: int,
    *,
    cpu_count: int | None = None,
    minimum_items_per_worker: int = 20_000,
    max_workers: int = 8,
) -> int:
    """Pick a worker count for item serialization.

    Serializing an item is pure Python — normalize, canonicalize, format — and
    it does not get faster on a bigger GPU. At competition scale it is tens of
    seconds of single-threaded work sitting on the critical path.

    Capped at 8 because each worker decodes a whole parquet row group including
    the attributes column. Against the 13M-row full item file that is roughly a
    gigabyte per worker, which is how a production refit got OOM-killed with
    exit 137. Callers must opt in explicitly; the default stays serial.
    """
    override = os.environ.get("ECUP_V7_SERIALIZE_WORKERS", "").strip()
    if override:
        try:
            return max(1, min(int(override), max_workers))
        except ValueError:
            pass
    detected = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    if detected <= 2 or requested_items < 2 * minimum_items_per_worker:
        return 1
    by_size = max(1, requested_items // minimum_items_per_worker)
    return max(1, min(detected - 1, by_size, max_workers))


_WORKER_STATE: dict[str, object] = {}


def _serialize_row_groups(groups: tuple[int, ...]) -> list[tuple[object, str, str]]:
    state = _WORKER_STATE
    parquet = pq.ParquetFile(str(state["path"]))
    columns = list(state["columns"])
    value_set = state["value_set"]
    max_chars = int(state["max_chars"])
    importance = state["attribute_importance"]
    out: list[tuple[object, str, str]] = []
    for group in groups:
        table = parquet.read_row_group(group, columns=columns)
        selected = table.filter(pc.is_in(table.column("id"), value_set=value_set))
        if not selected.num_rows:
            del table, selected
            continue
        payload = selected.to_pydict()
        for item_id, name, attributes, category in zip(
            payload["id"], payload["name"], payload["attributes"], payload["category"]
        ):
            text, cat = item_text_v7(
                item_id,
                name,
                attributes,
                category,
                max_chars=max_chars,
                attribute_importance=importance,
            )
            out.append((item_id, text, cat))
        del table, selected, payload
    return out


def build_v7_text_cache_from_parquet(
    parquet_path: Path,
    item_ids: Iterable[object],
    *,
    max_chars: int,
    batch_size: int = 131_072,
    attribute_importance: Mapping[str, float] | None = None,
    workers: int | None = None,
) -> tuple[dict[object, str], dict[object, str]]:
    """Stream selected full item records and serialize them without a giant DataFrame.

    v7 depends on canonical typed attributes. The legacy
    ``include_attributes=False`` fast path intentionally replaced attributes by
    ``{}``; that is correct for name-only consumers but silently defeats the v7
    hypothesis. This scanner keeps the real attributes while retaining only the
    final serialized strings and category map in memory.

    Serialization is distributed over row groups when it is worth it. Each item
    is a pure function of its own record, so the result does not depend on how
    the work was divided; only the first record wins for a duplicated id, in
    parquet order, exactly as in the serial path.
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

    # Serial unless the caller opts in. Training scans the full 13M-row item
    # file, where forked workers holding a row group each are what triggered the
    # OOM; the submission path scans only the test items and opts in.
    worker_count = 1 if workers is None else max(1, int(workers))
    row_groups = parquet.num_row_groups
    use_parallel = (
        worker_count > 1
        and row_groups > 1
        and "fork" in multiprocessing.get_all_start_methods()
        and os.environ.get("ECUP_V7_FORCE_SERIAL_SERIALIZE", "").strip() != "1"
    )
    if use_parallel:
        # Row groups are handed out in order and results are merged in that same
        # order, so a duplicated id resolves to the same record as the serial scan.
        shards: list[tuple[int, ...]] = [
            tuple(range(start, min(start + 1, row_groups))) for start in range(row_groups)
        ]
        print(
            f'{{"phase": "serialize-items-pool", "workers": {min(worker_count, len(shards))}, '
            f'"row_groups": {row_groups}, "requested_items": {len(requested)}}}',
            flush=True,
        )
        _WORKER_STATE.update(
            path=str(parquet_path),
            columns=columns,
            value_set=value_set,
            max_chars=int(max_chars),
            attribute_importance=attribute_importance,
        )
        try:
            context = multiprocessing.get_context("fork")
            progress = ProgressReporter(
                "serialize-items-parallel",
                len(requested),
                every_units=25_000,
                every_seconds=30.0,
            )
            with context.Pool(processes=min(worker_count, len(shards))) as pool:
                for records in pool.imap(_serialize_row_groups, shards):
                    for item_id, text, category in records:
                        if item_id in texts:
                            continue
                        texts[item_id] = text
                        categories[item_id] = category
                    progress.update(len(texts), workers=worker_count)
            progress.finish(len(texts), workers=worker_count)
        except OSError as error:
            print(
                f"[v7] serialization pool unavailable ({error}); falling back to serial",
                flush=True,
            )
            texts.clear()
            categories.clear()
            use_parallel = False
        finally:
            _WORKER_STATE.clear()

    if not use_parallel:
        progress = ProgressReporter(
            "serialize-items", len(requested), every_units=25_000, every_seconds=30.0
        )
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
            progress.update(len(texts))
            if len(texts) == len(requested):
                break
        progress.finish(len(texts))
    missing_ids = requested - set(texts)
    if missing_ids:
        first = min(missing_ids, key=lambda value: (type(value).__name__, repr(value)))
        raise KeyError(
            f"items parquet is missing {len(missing_ids)} requested IDs; first={first!r}"
        )
    del value_set
    pa.default_memory_pool().release_unused()
    return texts, categories


class _BatchPrefetcher:
    """Tokenize one batch ahead of the model on a background thread.

    Only ever one batch in flight, so peak memory is bounded. ``get`` verifies
    the bounds it is handed match what the producer built, and falls back to
    tokenizing inline if they ever disagree, so a mismatch can slow the run down
    but cannot silently score the wrong rows.
    """

    def __init__(self, tokenize, total_rows: int, batch_size: int, *, start: int = 0):
        self._tokenize = tokenize
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        bounds = []
        row = start
        while row < total_rows:
            stop = min(total_rows, row + batch_size)
            bounds.append((row, stop))
            row = stop
        self._thread = threading.Thread(
            target=self._produce, args=(bounds,), daemon=True, name="v7-tokenize"
        )
        self._thread.start()

    def _produce(self, bounds) -> None:
        for start, stop in bounds:
            if self._stop.is_set():
                break
            try:
                payload = (start, stop, self._tokenize(start, stop))
            except BaseException as error:  # surfaced to the consumer below
                payload = (start, stop, error)
            while not self._stop.is_set():
                try:
                    self._queue.put(payload, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def get(self, start: int, stop: int):
        while not self._stop.is_set():
            try:
                got_start, got_stop, payload = self._queue.get(timeout=5.0)
            except queue.Empty:
                break
            if (got_start, got_stop) != (start, stop):
                continue
            if isinstance(payload, BaseException):
                raise payload
            return payload
        return self._tokenize(start, stop)

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5.0)


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

    left_ids = frame["id1"].to_numpy()
    right_ids = frame["id2"].to_numpy()

    def tokenize(start: int, stop: int):
        return tokenizer(
            [texts[item_id] for item_id in left_ids[start:stop]],
            [texts[item_id] for item_id in right_ids[start:stop]],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    # Tokenization is CPU work that a faster accelerator does not shrink, so it
    # is produced one batch ahead of the model. Fast tokenizers release the GIL,
    # so a plain thread really does overlap with the forward pass. Batches stay
    # in the same order with the same contents, so scores are unchanged.
    prefetch = _BatchPrefetcher(tokenize, len(frame), current_batch)
    progress = ProgressReporter(
        "score-pairs", len(frame), every_units=20_000, every_seconds=30.0
    )
    with torch.inference_mode():
        while row < len(frame):
            stop = min(len(frame), row + current_batch)
            try:
                tokens = prefetch.get(row, stop)
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
                progress.update(row, batch_size=current_batch)
            except torch.cuda.OutOfMemoryError:
                if not on_cuda or current_batch <= 1:
                    prefetch.close()
                    raise
                current_batch = max(1, current_batch // 2)
                torch.cuda.empty_cache()
                # The queued batches were built for the old size; rebuild from
                # the row that failed so the sequence stays contiguous.
                prefetch.close()
                prefetch = _BatchPrefetcher(tokenize, len(frame), current_batch, start=row)
                print(
                    {"phase": "predict", "cuda_oom_batch_halved_to": current_batch},
                    flush=True,
                )
    prefetch.close()
    progress.finish(len(frame), batch_size=current_batch)
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
