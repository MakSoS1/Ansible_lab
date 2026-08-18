from ecup_matching.ml.run_v20_probe import candidate_flags


def test_candidate_modes_add_one_mechanism_at_a_time():
    assert candidate_flags("control") == {"generated": False, "rationale": False, "replay": False}
    assert candidate_flags("data-only") == {"generated": True, "rationale": False, "replay": False}
    assert candidate_flags("rationale") == {"generated": True, "rationale": True, "replay": False}
    assert candidate_flags("replay-data") == {"generated": True, "rationale": False, "replay": True}
    assert candidate_flags("replay-rationale") == {"generated": True, "rationale": True, "replay": True}
