from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


_REQUIRED_SOURCES: tuple[str, ...] = ("category", "weak", "sparse", "explicit")
_ORTHOGONAL_BASE: tuple[str, ...] = ("weak", "sparse", "explicit", "contrastive")
_ORTHOGONAL_EXTRAS: tuple[str, ...] = ("teacher2_raw", "weighted", "pretrained_raw")
_FIVE_SIGNAL_NAMES: tuple[str, ...] = ("weak", "sparse", "explicit", "contrastive", "teacher")


def _finite_1d(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_named_sources(
    scores: Mapping[str, object],
    required_names: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], int]:
    missing = [name for name in required_names if name not in scores]
    if missing:
        raise ValueError(f"missing required score sources: {missing}")
    arrays = {name: _finite_1d(scores[name], name=name) for name in required_names}
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("all score sources must have equal length")
    row_count = next(iter(lengths))
    if row_count == 0:
        raise ValueError("score sources must not be empty")
    return arrays, row_count


def _validated_sources(scores: Mapping[str, object]) -> tuple[dict[str, np.ndarray], int]:
    return _validated_named_sources(scores, _REQUIRED_SOURCES)


def percentile_rank(values) -> np.ndarray:
    """Return average percentile ranks in [0, 1] without using labels."""
    array = _finite_1d(values, name="values")
    if len(array) == 0:
        raise ValueError("values must not be empty")
    if len(array) == 1:
        return np.array([0.5], dtype=np.float64)
    raw_rank = pd.Series(array).rank(method="average").to_numpy(dtype=np.float64)
    return (raw_rank - 1.0) / float(len(array) - 1)


def grouped_percentile_rank(values, groups) -> np.ndarray:
    """Rank values independently inside each unlabeled scoring group."""
    array = _finite_1d(values, name="values")
    group_array = np.asarray(groups).astype(str)
    if group_array.ndim != 1 or len(group_array) != len(array):
        raise ValueError("values and groups must have equal length one-dimensional arrays")
    if len(array) == 0:
        raise ValueError("values must not be empty")

    result = np.empty(len(array), dtype=np.float64)
    for group in np.unique(group_array):
        mask = group_array == group
        result[mask] = percentile_rank(array[mask])
    return result


def fixed_blend_candidates(
    scores: Mapping[str, object],
    *,
    contrastive_cosine=None,
) -> dict[str, np.ndarray]:
    """Build the original predeclared target-free fusion candidates."""
    arrays, row_count = _validated_sources(scores)
    clipped = {name: np.clip(values, 1e-6, 1.0 - 1e-6) for name, values in arrays.items()}
    ranks = {name: percentile_rank(values) for name, values in arrays.items()}

    result: dict[str, np.ndarray] = {
        "prob_mean_4": np.mean(np.vstack([clipped[name] for name in _REQUIRED_SOURCES]), axis=0),
        "rank_mean_3": np.mean(
            np.vstack([ranks[name] for name in ("weak", "sparse", "explicit")]), axis=0
        ),
        "rank_mean_4": np.mean(np.vstack([ranks[name] for name in _REQUIRED_SOURCES]), axis=0),
    }

    if contrastive_cosine is not None:
        cosine = _finite_1d(contrastive_cosine, name="contrastive_cosine")
        if len(cosine) != row_count:
            raise ValueError("contrastive_cosine and score sources must have equal length")
        result["rank_mean_5"] = np.mean(
            np.vstack([ranks[name] for name in _REQUIRED_SOURCES] + [percentile_rank(cosine)]),
            axis=0,
        )
    return result


def rank_ablation_candidates(
    scores: Mapping[str, object],
    *,
    groups,
    contrastive_cosine,
) -> dict[str, np.ndarray]:
    """Build the bounded second target-free rank ablation declared in the design spec."""
    arrays, row_count = _validated_sources(scores)
    group_array = np.asarray(groups).astype(str)
    if group_array.ndim != 1 or len(group_array) != row_count:
        raise ValueError("groups and score sources must have equal length")
    cosine = _finite_1d(contrastive_cosine, name="contrastive_cosine")
    if len(cosine) != row_count:
        raise ValueError("contrastive_cosine and score sources must have equal length")

    global_ranks = {name: percentile_rank(values) for name, values in arrays.items()}
    global_cosine = percentile_rank(cosine)
    grouped_ranks = {
        name: grouped_percentile_rank(values, group_array) for name, values in arrays.items()
    }
    grouped_cosine = grouped_percentile_rank(cosine, group_array)

    return {
        "global_rank_mean_4_no_category": np.mean(
            np.vstack(
                [
                    global_ranks["weak"],
                    global_ranks["sparse"],
                    global_ranks["explicit"],
                    global_cosine,
                ]
            ),
            axis=0,
        ),
        "global_rank_mean_3_strong": np.mean(
            np.vstack([global_ranks["sparse"], global_ranks["explicit"], global_cosine]),
            axis=0,
        ),
        "category_rank_mean_5": np.mean(
            np.vstack(
                [
                    grouped_ranks["category"],
                    grouped_ranks["weak"],
                    grouped_ranks["sparse"],
                    grouped_ranks["explicit"],
                    grouped_cosine,
                ]
            ),
            axis=0,
        ),
        "category_rank_mean_4_no_category": np.mean(
            np.vstack(
                [
                    grouped_ranks["weak"],
                    grouped_ranks["sparse"],
                    grouped_ranks["explicit"],
                    grouped_cosine,
                ]
            ),
            axis=0,
        ),
        "category_rank_mean_3_strong": np.mean(
            np.vstack([grouped_ranks["sparse"], grouped_ranks["explicit"], grouped_cosine]),
            axis=0,
        ),
    }


def orthogonal_rank_candidates(
    current4_scores: Mapping[str, object],
    extra_scores: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Build the five predeclared global rank fusions from the orthogonal-evidence spec."""
    current4, row_count = _validated_named_sources(current4_scores, _ORTHOGONAL_BASE)
    extras, extra_count = _validated_named_sources(extra_scores, _ORTHOGONAL_EXTRAS)
    if extra_count != row_count:
        raise ValueError("all score sources must have equal length")

    ranked = {
        **{name: percentile_rank(values) for name, values in current4.items()},
        **{name: percentile_rank(values) for name, values in extras.items()},
    }

    def mean_rank(names: tuple[str, ...]) -> np.ndarray:
        return np.mean(np.vstack([ranked[name] for name in names]), axis=0)

    return {
        "current4_plus_teacher": mean_rank(_ORTHOGONAL_BASE + ("teacher2_raw",)),
        "current4_plus_weighted": mean_rank(_ORTHOGONAL_BASE + ("weighted",)),
        "current4_plus_pretrained": mean_rank(_ORTHOGONAL_BASE + ("pretrained_raw",)),
        "current4_plus_teacher_weighted": mean_rank(
            _ORTHOGONAL_BASE + ("teacher2_raw", "weighted")
        ),
        "current4_plus_all_three": mean_rank(_ORTHOGONAL_BASE + _ORTHOGONAL_EXTRAS),
    }


def leave_one_out_rank_candidates(five_signal_scores: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Build exactly the five leave-one-signal-out equal-rank robustness candidates."""
    arrays, _ = _validated_named_sources(five_signal_scores, _FIVE_SIGNAL_NAMES)
    ranked = {name: percentile_rank(values) for name, values in arrays.items()}

    def mean_without(drop: str) -> np.ndarray:
        names = [name for name in _FIVE_SIGNAL_NAMES if name != drop]
        return np.mean(np.vstack([ranked[name] for name in names]), axis=0)

    return {
        "loo_drop_weak": mean_without("weak"),
        "loo_drop_sparse": mean_without("sparse"),
        "loo_drop_explicit": mean_without("explicit"),
        "loo_drop_contrastive": mean_without("contrastive"),
        "loo_drop_teacher": mean_without("teacher"),
    }


def typed_explicit_rank_candidates(
    current5_scores: Mapping[str, object],
    typed_explicit_scores,
) -> dict[str, np.ndarray]:
    """Build exactly the add/replace typed-explicit candidates frozen before training completed."""
    current5, row_count = _validated_named_sources(current5_scores, _FIVE_SIGNAL_NAMES)
    typed = _finite_1d(typed_explicit_scores, name="typed_explicit")
    if len(typed) != row_count:
        raise ValueError("typed_explicit and current five score sources must have equal length")
    ranked = {name: percentile_rank(values) for name, values in current5.items()}
    typed_rank = percentile_rank(typed)

    add = np.mean(
        np.vstack([ranked[name] for name in _FIVE_SIGNAL_NAMES] + [typed_rank]),
        axis=0,
    )
    replace_names = ("weak", "sparse", "contrastive", "teacher")
    replace = np.mean(
        np.vstack([ranked[name] for name in replace_names] + [typed_rank]),
        axis=0,
    )
    return {
        "current5_plus_typed_explicit": add,
        "current5_replace_explicit_with_typed": replace,
    }


def normalized_retrain_rank_candidates(
    base5_scores: Mapping[str, object],
    *,
    typed_explicit_scores,
    normalized_explicit_scores,
    normalized_category_scores,
) -> dict[str, np.ndarray]:
    """Build the four target-free normalized-retrain candidates frozen before retrains finished."""
    base5, row_count = _validated_named_sources(base5_scores, _FIVE_SIGNAL_NAMES)
    typed = _finite_1d(typed_explicit_scores, name="typed_explicit")
    normalized_explicit = _finite_1d(normalized_explicit_scores, name="normalized_explicit")
    normalized_category = _finite_1d(normalized_category_scores, name="normalized_category")
    if not (len(typed) == len(normalized_explicit) == len(normalized_category) == row_count):
        raise ValueError("normalized retrain score sources must have equal length")

    ranks = {name: percentile_rank(values) for name, values in base5.items()}
    typed_rank = percentile_rank(typed)
    normalized_explicit_rank = percentile_rank(normalized_explicit)
    normalized_category_rank = percentile_rank(normalized_category)

    current6_ranks = [ranks[name] for name in _FIVE_SIGNAL_NAMES] + [typed_rank]
    add_explicit = np.mean(np.vstack(current6_ranks + [normalized_explicit_rank]), axis=0)
    replace_typed = np.mean(
        np.vstack([ranks[name] for name in _FIVE_SIGNAL_NAMES] + [normalized_explicit_rank]),
        axis=0,
    )
    replace_old = np.mean(
        np.vstack(
            [ranks[name] for name in ("weak", "sparse", "contrastive", "teacher")]
            + [normalized_explicit_rank, typed_rank]
        ),
        axis=0,
    )
    add_category = np.mean(np.vstack(current6_ranks + [normalized_category_rank]), axis=0)

    return {
        "current6_plus_normalized_explicit": add_explicit,
        "current6_replace_typed_with_normalized_explicit": replace_typed,
        "current6_replace_old_explicit_with_normalized_explicit": replace_old,
        "current6_plus_normalized_category": add_category,
    }
