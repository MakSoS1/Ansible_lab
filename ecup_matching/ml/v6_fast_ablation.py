from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .v5_fixed_blend import percentile_rank


# Keep the production ordering local. Importing v5_meta_blend only for this
# constant dragged its training-only sklearn metric dependency into submission
# packages; the tuple itself is part of the frozen v5/v6 runtime contract.
SIX_SIGNAL_NAMES: tuple[str, ...] = (
    "weak",
    "sparse",
    "explicit",
    "contrastive",
    "teacher",
    "typed_explicit",
)


@dataclass(frozen=True)
class FastCandidateSpec:
    retained_signals: tuple[str, ...]
    required_expensive_signals: tuple[str, ...]
    surrogate_targets: tuple[str, ...]


CANDIDATE_SPECS: dict[str, FastCandidateSpec] = {
    "structured_only": FastCandidateSpec(
        retained_signals=("weak", "sparse", "explicit", "typed_explicit"),
        required_expensive_signals=(),
        surrogate_targets=("contrastive", "teacher"),
    ),
    "no_teacher": FastCandidateSpec(
        retained_signals=("weak", "sparse", "explicit", "contrastive", "typed_explicit"),
        required_expensive_signals=("contrastive",),
        surrogate_targets=("teacher",),
    ),
    "no_contrastive": FastCandidateSpec(
        retained_signals=("weak", "sparse", "explicit", "teacher", "typed_explicit"),
        required_expensive_signals=("teacher",),
        surrogate_targets=("contrastive",),
    ),
}


def _finite_1d(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional score vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def target_free_rank_surrogate(
    scores: Mapping[str, object],
    source_names: tuple[str, ...],
) -> np.ndarray:
    if not source_names:
        raise ValueError("source_names must not be empty")
    ranked: list[np.ndarray] = []
    expected_length: int | None = None
    for name in source_names:
        if name not in scores:
            raise ValueError(f"missing retained signal: {name}")
        values = _finite_1d(scores[name], name=name)
        if expected_length is None:
            expected_length = len(values)
        elif len(values) != expected_length:
            raise ValueError("retained signal lengths do not match")
        ranked.append(percentile_rank(values))
    return np.mean(np.column_stack(ranked), axis=1, dtype=np.float64)


def build_fast_candidate_scores(
    scores: Mapping[str, object],
    candidate_name: str,
) -> dict[str, np.ndarray]:
    if candidate_name not in CANDIDATE_SPECS:
        raise ValueError(f"unknown fast candidate: {candidate_name}")
    spec = CANDIDATE_SPECS[candidate_name]

    retained: dict[str, np.ndarray] = {}
    expected_length: int | None = None
    for name in spec.retained_signals:
        if name not in scores:
            raise ValueError(f"missing retained signal: {name}")
        values = _finite_1d(scores[name], name=name)
        if expected_length is None:
            expected_length = len(values)
        elif len(values) != expected_length:
            raise ValueError("retained signal lengths do not match")
        retained[name] = values

    surrogate = target_free_rank_surrogate(retained, spec.retained_signals)
    result: dict[str, np.ndarray] = {}
    for name in SIX_SIGNAL_NAMES:
        if name in retained:
            result[name] = retained[name]
        elif name in spec.surrogate_targets:
            result[name] = surrogate.copy()
        else:
            raise RuntimeError(f"candidate {candidate_name!r} has no source for {name!r}")
    return result
