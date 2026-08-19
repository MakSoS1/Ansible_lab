from __future__ import annotations

from ecup_matching.ml.v21_selection import select_provisional_keeper, v20_metrics_to_transfer


def _v20(*, human: float, weak: float, brier: float, proxy: float, tail: float) -> dict[str, object]:
    return {
        "human_macro_average_precision": human,
        "human_per_category_ap": {"A": human, "B": human - 0.01},
        "human_tail_macro_average_precision": tail,
        "weak_metrics": {"macro_average_precision": weak, "soft_brier": brier},
        "proxy_metrics": {"macro_average_precision": proxy},
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "cross_split_item_overlap": 0,
        "held_rows": 1000,
    }


def _anchors() -> dict[str, float]:
    return {"v7": 0.500, "v13b": 0.520, "v12": 0.530, "v14": 0.540, "v19": 0.640}


def test_v20_metric_adapter_preserves_transfer_axes() -> None:
    got = v20_metrics_to_transfer(_v20(human=.70, weak=.65, brier=.11, proxy=.70, tail=.68))
    assert got["human_macro_average_precision"] == .70
    assert got["weak_macro_average_precision"] == .65
    assert got["weak_soft_brier"] == .11
    assert got["tail_macro_average_precision"] == .68
    assert got["category_row_counts"] == {"A": 1000, "B": 1000}


def test_provisional_selection_ranks_safe_candidates_by_public_extrapolation() -> None:
    control = _v20(human=.700, weak=.650, brier=.110, proxy=.60, tail=.680)
    candidates = {
        "data-only": _v20(human=.701, weak=.660, brier=.109, proxy=.72, tail=.681),
        "rationale": _v20(human=.700, weak=.662, brier=.109, proxy=.83, tail=.682),
    }
    result = select_provisional_keeper(control, candidates, _anchors())
    assert result["selected"] == "rationale"
    assert result["evaluations"]["rationale"]["safe"] is True
    assert result["evaluations"]["rationale"]["calibration"]["proxy_implied_public_lb"] > result["evaluations"]["data-only"]["calibration"]["proxy_implied_public_lb"]


def test_unsafe_high_proxy_candidate_is_never_selected() -> None:
    control = _v20(human=.700, weak=.650, brier=.110, proxy=.60, tail=.680)
    candidates = {
        "safe": _v20(human=.701, weak=.660, brier=.109, proxy=.72, tail=.681),
        "unsafe": _v20(human=.690, weak=.670, brier=.109, proxy=.99, tail=.681),
    }
    result = select_provisional_keeper(control, candidates, _anchors())
    assert result["selected"] == "safe"
    assert result["evaluations"]["unsafe"]["safe"] is False
