from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Collection, Hashable

import pandas as pd


@dataclass(frozen=True)
class V7TeacherConfig:
    max_length: int
    curriculum_rows: int
    effective_batch_size: int
    epochs: float
    max_steps: int | None = None


def required_optimizer_steps(config: V7TeacherConfig) -> int:
    if config.curriculum_rows <= 0:
        raise ValueError("curriculum_rows must be positive")
    if config.effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    if not math.isfinite(float(config.epochs)) or float(config.epochs) <= 0:
        raise ValueError("epochs must be finite and positive")
    steps_per_epoch = math.ceil(config.curriculum_rows / config.effective_batch_size)
    return int(math.ceil(steps_per_epoch * float(config.epochs)))


def validate_v7_teacher_config(config: V7TeacherConfig) -> V7TeacherConfig:
    if config.max_length < 256:
        raise ValueError("v7 max_length must be at least 256")
    needed = required_optimizer_steps(config)
    if config.max_steps is not None:
        if config.max_steps <= 0:
            raise ValueError("max_steps must be positive when specified")
        if int(config.max_steps) < needed:
            raise ValueError(
                f"v7 optimizer steps are truncated: max_steps={config.max_steps}, required={needed}"
            )
    return config


def filter_forbidden_weak_pairs(
    weak: pd.DataFrame,
    *,
    forbidden_item_ids: Collection[Hashable],
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"id1", "id2"}
    missing = required - set(weak.columns)
    if missing:
        raise ValueError(f"weak pairs missing columns: {sorted(missing)}")
    forbidden = set(forbidden_item_ids)
    mask = ~weak["id1"].isin(forbidden) & ~weak["id2"].isin(forbidden)
    kept = weak.loc[mask].copy().reset_index(drop=True)
    report = {
        "input_rows": int(len(weak)),
        "removed_rows": int((~mask).sum()),
        "kept_rows": int(mask.sum()),
    }
    endpoints = set(kept["id1"]) | set(kept["id2"])
    if endpoints & forbidden:
        raise RuntimeError("forbidden item survived weak-pair filtering")
    return kept, report
