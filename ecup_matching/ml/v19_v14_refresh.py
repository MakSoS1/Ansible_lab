from __future__ import annotations

from typing import Mapping

import pandas as pd

from .v19_refresh_gate import evaluate_refresh


def filter_refresh_pairs(
    candidate: pd.DataFrame,
    seen: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove candidate rows touching any endpoint from a previously seen weak corpus.

    The function deliberately uses endpoint-level exclusion rather than exact-pair
    exclusion.  This creates a more conservative refresh corpus: neither product
    card in a retained refresh pair appeared in the reconstructed historical weak
    training slice.
    """
    required_candidate = {"id1", "id2", "target"}
    required_seen = {"id1", "id2"}
    missing_candidate = required_candidate - set(candidate.columns)
    missing_seen = required_seen - set(seen.columns)
    if missing_candidate:
        raise ValueError(f"candidate refresh pairs missing columns: {sorted(missing_candidate)}")
    if missing_seen:
        raise ValueError(f"seen weak pairs missing columns: {sorted(missing_seen)}")
    seen_endpoints = set(seen["id1"].tolist()) | set(seen["id2"].tolist())
    mask = ~candidate["id1"].isin(seen_endpoints) & ~candidate["id2"].isin(seen_endpoints)
    out = candidate.loc[mask].copy().reset_index(drop=True)
    return out, {
        "input_rows": int(len(candidate)),
        "output_rows": int(len(out)),
        "seen_endpoint_count": int(len(seen_endpoints)),
        "removed_seen_endpoint_rows": int((~mask).sum()),
    }


def select_refresh_keeper(
    baseline: Mapping[str, object],
    candidates: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Apply the existing v19 anti-forgetting gate and choose the strongest pass.

    Keeper ordering is intentionally simple and preregisterable: maximize weak
    AP gain first, then human-retention delta, then candidate name for stable
    deterministic tie-breaking.  A candidate that does not pass the existing
    v19 weak/human/category/Brier gate can never be selected.
    """
    evaluations: dict[str, dict[str, object]] = {}
    passing: list[tuple[float, float, str]] = []
    for name, metrics in sorted(candidates.items()):
        result = evaluate_refresh(baseline, metrics)
        evaluations[str(name)] = result
        if result["promote"]:
            passing.append(
                (
                    float(result["weak_delta"]),
                    float(result["human_delta"]),
                    str(name),
                )
            )
    keeper = None
    if passing:
        keeper = max(passing, key=lambda value: (value[0], value[1], value[2]))[2]
    return {
        "version": "v19-v14-refresh-selection-v1",
        "keeper": keeper,
        "evaluations": evaluations,
        "no_keeper": keeper is None,
        "rule": "existing v19 gate; maximize weak_delta then human_delta",
    }


__all__ = ["filter_refresh_pairs", "select_refresh_keeper"]
