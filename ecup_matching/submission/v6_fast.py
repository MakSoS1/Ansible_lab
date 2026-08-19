from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


GIB = 1024**3


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    contrastive_batch: int
    teacher_batch: int
    autocast_dtype: str | None
    non_blocking: bool


def select_runtime_config(*, total_memory_bytes: int, device_type: str) -> RuntimeConfig:
    if not isinstance(total_memory_bytes, int) or total_memory_bytes < 0:
        raise ValueError("total_memory_bytes must be a non-negative integer")
    device = str(device_type).strip().lower()
    if device == "cpu":
        return RuntimeConfig(
            device="cpu",
            contrastive_batch=32,
            teacher_batch=16,
            autocast_dtype=None,
            non_blocking=False,
        )
    if device != "cuda":
        raise ValueError("device_type must be 'cpu' or 'cuda'")

    if total_memory_bytes >= 60 * GIB:
        contrastive_batch = 1024
        teacher_batch = 384
    elif total_memory_bytes >= 20 * GIB:
        contrastive_batch = 512
        teacher_batch = 192
    else:
        contrastive_batch = 256
        teacher_batch = 96
    return RuntimeConfig(
        device="cuda",
        contrastive_batch=contrastive_batch,
        teacher_batch=teacher_batch,
        autocast_dtype="float16",
        non_blocking=True,
    )


def batch_index_ranges(row_count: int, batch_size: int) -> Iterator[tuple[int, int]]:
    if not isinstance(row_count, int) or row_count < 0:
        raise ValueError("row_count must be a non-negative integer")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, row_count, batch_size):
        yield start, min(start + batch_size, row_count)


def collect_chunked_scores(
    *,
    row_count: int,
    chunk_size: int,
    signal_names: Sequence[str],
    score_chunk: Callable[[int, int], Mapping[str, object] | object],
) -> dict[str, np.ndarray]:
    """Collect aligned score vectors while bounding temporary pair-level state.

    ``score_chunk`` receives half-open global row positions and must return one
    finite one-dimensional vector per requested signal, with exactly
    ``end-start`` values. A raw vector is accepted only when one signal is
    requested. The returned arrays preserve the original global row order.
    """
    names = tuple(str(name) for name in signal_names)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("signal_names must contain unique non-empty names")
    buffers = {
        name: np.empty(row_count, dtype=np.float64)
        for name in names
    }
    for start, end in batch_index_ranges(row_count, chunk_size):
        payload = score_chunk(start, end)
        if isinstance(payload, Mapping):
            missing = [name for name in names if name not in payload]
            if missing:
                raise ValueError(f"missing score signals: {missing}")
            unexpected = [name for name in payload if name not in buffers]
            if unexpected:
                raise ValueError(f"unexpected score signals: {unexpected}")
            chunk_scores = payload
        elif len(names) == 1:
            chunk_scores = {names[0]: payload}
        else:
            raise ValueError("score_chunk must return a mapping for multiple signals")

        expected = end - start
        for name in names:
            values = np.asarray(chunk_scores[name], dtype=np.float64)
            if values.ndim != 1 or len(values) != expected:
                raise ValueError(
                    f"score signal {name!r} must contain exactly {expected} values"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"score signal {name!r} must be finite")
            buffers[name][start:end] = values
    return buffers


def move_token_batch(tokens: Mapping[str, Any], config: RuntimeConfig) -> dict[str, Any]:
    return {
        name: value.to(config.device, non_blocking=config.non_blocking)
        for name, value in tokens.items()
    }


def torch_autocast(torch_module: Any, config: RuntimeConfig):
    if config.autocast_dtype is None:
        return contextlib.nullcontext()
    if config.autocast_dtype != "float16":
        raise ValueError(f"unsupported autocast dtype: {config.autocast_dtype}")
    return torch_module.autocast(device_type=config.device, dtype=torch_module.float16)


def aligned_pair_texts(
    left_ids: Sequence[Any],
    right_ids: Sequence[Any],
    text_by_id: Mapping[Any, str],
) -> tuple[list[str], list[str]]:
    if len(left_ids) != len(right_ids):
        raise ValueError("left_ids and right_ids must have equal length")
    return (
        [text_by_id[item_id] for item_id in left_ids],
        [text_by_id[item_id] for item_id in right_ids],
    )
