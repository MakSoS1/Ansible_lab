from ecup_matching.ml.run_v4_training import submission_eligible


def test_submission_eligible_requires_strict_improvement_and_neural_route() -> None:
    assert submission_eligible(
        {"accepted_as_improvement": True},
        {"category_alphas": {"__global__": 0.55}},
    ) is True

    assert submission_eligible(
        {"accepted_as_improvement": False},
        {"category_alphas": {"__global__": 0.55}},
    ) is False

    assert submission_eligible(
        {"accepted_as_improvement": True},
        {"category_alphas": {"__global__": 0.0}},
    ) is False

    assert submission_eligible(
        {"accepted_as_improvement": True},
        {"category_alphas": {"electronics": 0.0, "jewelry": 0.2}},
    ) is True
