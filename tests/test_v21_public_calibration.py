from __future__ import annotations

import pytest

from ecup_matching.ml.v21_public_calibration import (
    EXPECTED_PUBLIC_ORDER,
    HISTORICAL_PUBLIC_LB,
    TARGET_PUBLIC_LB,
    calibrate_candidate_public_lb,
    validate_anchor_proxy,
)


def _anchors() -> dict[str, float]:
    # Synthetic proxy scores with the same required historical ordering.
    return {
        "v7": 0.500,
        "v13b": 0.520,
        "v12": 0.530,
        "v14": 0.540,
        "v19": 0.640,
    }


def test_v21_registers_empirical_v19_anchor_and_public_order() -> None:
    assert HISTORICAL_PUBLIC_LB["v19"] == pytest.approx(0.41)
    assert HISTORICAL_PUBLIC_LB["v14"] == pytest.approx(0.38032704703111925)
    assert EXPECTED_PUBLIC_ORDER == ("v19", "v14", "v12", "v13b", "v7")
    assert TARGET_PUBLIC_LB == pytest.approx(0.5)


def test_anchor_validation_is_fail_closed() -> None:
    report = validate_anchor_proxy(_anchors())
    assert report["calibrated"] is True
    assert report["best_anchor"] == "v19"

    missing = _anchors()
    missing.pop("v19")
    with pytest.raises(ValueError):
        validate_anchor_proxy(missing)

    inverted = _anchors()
    inverted["v13b"] = 0.535
    with pytest.raises(ValueError):
        validate_anchor_proxy(inverted)


def test_target_half_requires_material_gap_above_v19() -> None:
    anchors = _anchors()
    # v14->v19 proxy gap is .10. Public 0.41->0.50 requires ~3.033x that
    # additional proxy gap under the deliberately low-capacity extrapolator.
    too_small = calibrate_candidate_public_lb(anchors, candidate_proxy=0.70)
    assert too_small["candidate_above_v19"] is True
    assert too_small["target_reached"] is False
    assert too_small["proxy_implied_public_lb"] < 0.5

    material = calibrate_candidate_public_lb(anchors, candidate_proxy=0.95)
    assert material["candidate_above_v19"] is True
    assert material["normalized_gap_above_v19"] > 3.03
    assert material["proxy_implied_public_lb"] >= 0.5
    assert material["target_reached"] is True


def test_calibration_rejects_nonfinite_or_nonpositive_anchor_gap() -> None:
    anchors = _anchors()
    anchors["v19"] = anchors["v14"]
    with pytest.raises(ValueError):
        calibrate_candidate_public_lb(anchors, candidate_proxy=0.9)
    with pytest.raises(ValueError):
        calibrate_candidate_public_lb(_anchors(), candidate_proxy=float("nan"))
