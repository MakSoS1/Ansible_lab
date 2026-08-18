import pandas as pd
import pytest

from ecup_matching.ml.v20_policy import (
    PUBLIC_LB_ANCHORS,
    V20Policy,
    policy_sha256,
    validate_fold_exclusion,
)


def test_v20_policy_is_frozen_to_preregistered_contract():
    p = V20Policy()
    assert p.positive_precision_lcb == 0.985
    assert p.negative_precision_lcb == 0.995
    assert p.category_precision_lcb == 0.970
    assert p.critical_precision_lcb == 0.950
    assert p.max_length == 256
    assert p.production_checkpoints == 1
    assert p.lambda_reason == 0.15
    assert p.lambda_consistency == 0.05
    assert p.phase_b_human_to_other == (1, 2)
    assert p.phase_c_human_to_other == (4, 1)


def test_public_anchors_match_external_evidence():
    assert list(PUBLIC_LB_ANCHORS) == ["v14", "v12", "v13B", "v7"]
    assert PUBLIC_LB_ANCHORS["v14"] > PUBLIC_LB_ANCHORS["v12"] > PUBLIC_LB_ANCHORS["v13B"] > PUBLIC_LB_ANCHORS["v7"]


def test_policy_hash_is_deterministic():
    assert policy_sha256(V20Policy()) == policy_sha256(V20Policy())
    assert len(policy_sha256(V20Policy())) == 64


def test_fold_exclusion_rejects_either_endpoint():
    frame = pd.DataFrame({"id1": [1, 2], "id2": [3, 4]})
    with pytest.raises(ValueError, match="held-fold item"):
        validate_fold_exclusion(frame, {4})
    assert validate_fold_exclusion(frame, {99}) == 0
