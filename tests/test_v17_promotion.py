from ecup_matching.ml.v17_promotion import evaluate_promotion


def test_promotes_only_when_both_fixed_gates_pass():
    result = evaluate_promotion(
        control_weak=0.40,
        scaled_weak=0.406,
        control_human=0.70,
        scaled_human=0.696,
    )
    assert result["weak_gate"] is True
    assert result["human_gate"] is True
    assert result["promote"] is True


def test_rejects_exact_weak_threshold_because_rule_is_strict():
    result = evaluate_promotion(
        control_weak=0.40,
        scaled_weak=0.405,
        control_human=0.70,
        scaled_human=0.70,
    )
    assert result["weak_gate"] is False
    assert result["promote"] is False


def test_rejects_human_drop_beyond_floor():
    result = evaluate_promotion(
        control_weak=0.40,
        scaled_weak=0.41,
        control_human=0.70,
        scaled_human=0.6949,
    )
    assert result["human_gate"] is False
    assert result["promote"] is False
