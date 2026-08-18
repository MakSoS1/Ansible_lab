from ecup_matching.ml.v20_select import select_stage1, select_replay


def metric(candidate, human, proxy, tail, cats):
    return {
        "candidate": candidate,
        "human_macro_average_precision": human,
        "human_per_category_ap": cats,
        "proxy_metrics": {"macro_average_precision": proxy, "tail_macro_average_precision": tail},
    }


def test_stage1_chooses_best_promoted_data_or_rationale():
    control = metric("control", 0.70, 0.50, 0.40, {"x": 0.5})
    data = metric("data-only", 0.701, 0.507, 0.405, {"x": 0.50})
    rationale = metric("rationale", 0.702, 0.509, 0.406, {"x": 0.51})
    out = select_stage1(control, {"data-only": data, "rationale": rationale}, proxy_promotable=True)
    assert out["selected"] == "rationale"


def test_replay_only_replaces_keeper_if_it_passes_and_improves_proxy():
    control = metric("control", 0.70, 0.50, 0.40, {"x": 0.5})
    keeper = metric("data-only", 0.701, 0.507, 0.405, {"x": 0.50})
    replay = metric("replay-data", 0.702, 0.514, 0.410, {"x": 0.51})
    out = select_replay(control, keeper, replay, proxy_promotable=True)
    assert out["selected"] == "replay-data"
