from ecup_matching.ml.v20_promotion import evaluate_candidate, evaluate_scaled_confirmation


def test_candidate_requires_strict_proxy_gain():
    result = evaluate_candidate(
        proxy_delta=0.005,
        human_delta=0.0,
        audited_tail_delta=0.0,
        category_deltas={"x": 0.0},
        proxy_axis_promotable=True,
    )
    assert result["promote"] is False
    assert result["proxy_gate"] is False


def test_candidate_allows_exact_human_floor_but_rejects_category_crash():
    ok = evaluate_candidate(
        proxy_delta=0.006,
        human_delta=-0.003,
        audited_tail_delta=-0.02,
        category_deltas={"x": -0.04, "y": 0.01},
        proxy_axis_promotable=True,
    )
    assert ok["promote"] is True
    bad = evaluate_candidate(
        proxy_delta=0.006,
        human_delta=0.0,
        audited_tail_delta=0.0,
        category_deltas={"x": -0.0401},
        proxy_axis_promotable=True,
    )
    assert bad["promote"] is False


def test_unpromotable_proxy_cannot_select_candidate():
    result = evaluate_candidate(
        proxy_delta=0.5,
        human_delta=0.1,
        audited_tail_delta=0.1,
        category_deltas={"x": 0.1},
        proxy_axis_promotable=False,
    )
    assert result["promote"] is False


def test_scaled_confirmation_requires_each_fold_and_nonnegative_mean_human():
    folds = [
        {"promote": True, "human_delta": -0.001},
        {"promote": True, "human_delta": 0.001},
    ]
    assert evaluate_scaled_confirmation(folds)["promote"] is True
    folds[1] = {"promote": True, "human_delta": 0.0005}
    assert evaluate_scaled_confirmation(folds)["promote"] is False
