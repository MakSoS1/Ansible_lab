from ecup_matching.ml.v20_select import select_stage1, select_replay


def metric(candidate, human, proxy, tail, cats):
    return {
        "candidate": candidate,
        "human_macro_average_precision": human,
        "human_per_category_ap": cats,
        "proxy_metrics": {"macro_average_precision": proxy, "tail_macro_average_precision": tail},
    }


def test_stage1_chooses_best_promoted_data_or_rationale_above_anchor():
    control = metric("control", 0.70, 0.50, 0.40, {"x": 0.5})
    data = metric("data-only", 0.701, 0.507, 0.405, {"x": 0.50})
    rationale = metric("rationale", 0.702, 0.509, 0.406, {"x": 0.51})
    out = select_stage1(
        control, {"data-only": data, "rationale": rationale},
        proxy_promotable=True, best_anchor_proxy=0.508,
    )
    assert out["selected"] == "rationale"


def test_stage1_rejects_candidates_that_do_not_beat_v14_proxy():
    control = metric("control", 0.70, 0.50, 0.40, {"x": 0.5})
    data = metric("data-only", 0.701, 0.507, 0.405, {"x": 0.50})
    rationale = metric("rationale", 0.702, 0.509, 0.406, {"x": 0.51})
    out = select_stage1(
        control, {"data-only": data, "rationale": rationale},
        proxy_promotable=True, best_anchor_proxy=0.510,
    )
    assert out["selected"] is None
    assert out["promote"] is False


def test_replay_only_replaces_keeper_if_it_passes_and_improves_proxy():
    control = metric("control", 0.70, 0.50, 0.40, {"x": 0.5})
    keeper = metric("data-only", 0.701, 0.507, 0.405, {"x": 0.50})
    replay = metric("replay-data", 0.702, 0.514, 0.410, {"x": 0.51})
    out = select_replay(
        control, keeper, replay, proxy_promotable=True, best_anchor_proxy=0.508,
    )
    assert out["selected"] == "replay-data"


def test_replay_cannot_replace_keeper_below_historical_anchor():
    control = metric("control", 0.70, 0.50, 0.40, {"x": 0.5})
    keeper = metric("data-only", 0.701, 0.512, 0.405, {"x": 0.50})
    replay = metric("replay-data", 0.702, 0.514, 0.410, {"x": 0.51})
    out = select_replay(
        control, keeper, replay, proxy_promotable=True, best_anchor_proxy=0.515,
    )
    assert out["selected"] == "data-only"
