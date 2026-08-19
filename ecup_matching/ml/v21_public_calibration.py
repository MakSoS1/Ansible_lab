from __future__ import annotations

import math
from typing import Mapping

HISTORICAL_PUBLIC_LB: dict[str, float] = {
    "v7": 0.3655833314,
    "v12": 0.379811620418641,
    "v13b": 0.37837816527590995,
    "v14": 0.38032704703111925,
    # Empirical Public LB reported after the canonical v19 submission.
    "v19": 0.41,
}
EXPECTED_PUBLIC_ORDER = ("v19", "v14", "v12", "v13b", "v7")
TARGET_PUBLIC_LB = 0.5
_EPS = 1e-12


def _finite(value: object, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def validate_anchor_proxy(anchor_proxy: Mapping[str, object]) -> dict[str, object]:
    missing = [name for name in EXPECTED_PUBLIC_ORDER if name not in anchor_proxy]
    if missing:
        raise ValueError(f"missing historical proxy anchors: {missing}")
    observed = {name: _finite(anchor_proxy[name], f"proxy[{name}]") for name in EXPECTED_PUBLIC_ORDER}
    ranked = tuple(sorted(observed, key=lambda name: (-observed[name], name)))
    if ranked != EXPECTED_PUBLIC_ORDER:
        raise ValueError(
            f"proxy does not reproduce empirical Public order: observed={ranked}, expected={EXPECTED_PUBLIC_ORDER}"
        )
    if observed["v19"] - observed["v14"] <= _EPS:
        raise ValueError("v19-v14 proxy gap must be strictly positive")
    return {
        "version": "v21-public-anchor-validation-v1",
        "calibrated": True,
        "best_anchor": "v19",
        "observed_order": ranked,
        "expected_order": EXPECTED_PUBLIC_ORDER,
        "anchor_proxy": observed,
        "historical_public_lb": dict(HISTORICAL_PUBLIC_LB),
    }


def calibrate_candidate_public_lb(
    anchor_proxy: Mapping[str, object],
    *,
    candidate_proxy: float,
    target_public_lb: float = TARGET_PUBLIC_LB,
) -> dict[str, object]:
    """Low-capacity extrapolation using only the newest observed transfer segment.

    Five Public observations are far too few to justify a flexible regression.  v21
    therefore uses the empirical v14->v19 Public gain as one fixed slope segment and
    expresses a candidate only as a multiple of the corresponding proxy gap.  This
    makes the >0.5 hypothesis explicit rather than silently overfitting five anchors.
    """
    validation = validate_anchor_proxy(anchor_proxy)
    proxy = dict(validation["anchor_proxy"])
    candidate = _finite(candidate_proxy, "candidate_proxy")
    target = _finite(target_public_lb, "target_public_lb")

    proxy_gap = float(proxy["v19"]) - float(proxy["v14"])
    public_gap = HISTORICAL_PUBLIC_LB["v19"] - HISTORICAL_PUBLIC_LB["v14"]
    if proxy_gap <= _EPS or public_gap <= _EPS:
        raise ValueError("calibration gaps must be strictly positive")

    normalized = (candidate - float(proxy["v19"])) / proxy_gap
    implied = HISTORICAL_PUBLIC_LB["v19"] + normalized * public_gap
    required = (target - HISTORICAL_PUBLIC_LB["v19"]) / public_gap
    above_v19 = candidate > float(proxy["v19"]) + _EPS
    target_reached = bool(above_v19 and normalized + _EPS >= required and implied + _EPS >= target)

    return {
        "version": "v21-public-gap-calibration-v1",
        "calibrated": True,
        "best_anchor": "v19",
        "candidate_proxy": float(candidate),
        "v19_proxy": float(proxy["v19"]),
        "v14_proxy": float(proxy["v14"]),
        "proxy_gap_v14_to_v19": float(proxy_gap),
        "public_gap_v14_to_v19": float(public_gap),
        "normalized_gap_above_v19": float(normalized),
        "required_normalized_gap_for_target": float(required),
        "proxy_implied_public_lb": float(implied),
        "target_public_lb": float(target),
        "candidate_above_v19": bool(above_v19),
        "target_reached": bool(target_reached),
        "anchor_validation": validation,
        "interpretation": "extrapolation hypothesis only; leaderboard result remains the final external measurement",
    }


__all__ = [
    "HISTORICAL_PUBLIC_LB",
    "EXPECTED_PUBLIC_ORDER",
    "TARGET_PUBLIC_LB",
    "validate_anchor_proxy",
    "calibrate_candidate_public_lb",
]
