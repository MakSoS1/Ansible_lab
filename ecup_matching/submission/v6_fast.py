from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


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
