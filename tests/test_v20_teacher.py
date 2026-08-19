import json

import pytest

from ecup_matching.ml.v20_teacher import TeacherDecision, consensus_label


def payload(verdict="MATCH", reason="SAME_MODEL"):
    return json.dumps({
        "verdict": verdict,
        "reason_code": reason,
        "same_product_type": True,
        "brand_left": "a",
        "brand_right": "a",
        "model_left": "x",
        "model_right": "x",
        "critical_attributes": {},
        "conflicts": [],
        "evidence": ["same model"],
    })


def test_malformed_teacher_json_is_rejected():
    with pytest.raises(ValueError):
        TeacherDecision.from_json("not json", teacher_id="t1", revision="r1", prompt_sha256="a" * 64)


def test_single_teacher_never_admits_new_llm_label():
    a = TeacherDecision.from_json(payload(), teacher_id="t1", revision="r1", prompt_sha256="a" * 64)
    result = consensus_label(a, None, deterministic_reason="SAME_MODEL")
    assert result["admitted"] is False
    assert result["reason"] == "two_teachers_required"


def test_disagreement_and_uncertain_are_rejected():
    a = TeacherDecision.from_json(payload("MATCH"), teacher_id="t1", revision="r1", prompt_sha256="a" * 64)
    b = TeacherDecision.from_json(payload("NON_MATCH", "MODEL_CONFLICT"), teacher_id="t2", revision="r2", prompt_sha256="a" * 64)
    assert consensus_label(a, b, deterministic_reason="SAME_MODEL")["admitted"] is False
    u = TeacherDecision.from_json(payload("UNCERTAIN", "SPARSE_EVIDENCE"), teacher_id="t2", revision="r2", prompt_sha256="a" * 64)
    assert consensus_label(a, u, deterministic_reason="SAME_MODEL")["admitted"] is False


def test_checker_conflict_is_rejected():
    a = TeacherDecision.from_json(payload(), teacher_id="t1", revision="r1", prompt_sha256="a" * 64)
    b = TeacherDecision.from_json(payload(), teacher_id="t2", revision="r2", prompt_sha256="a" * 64)
    assert consensus_label(a, b, deterministic_reason="MODEL_CONFLICT")["admitted"] is False


def test_consensus_accepts_and_records_provenance():
    a = TeacherDecision.from_json(payload(), teacher_id="t1", revision="r1", prompt_sha256="a" * 64)
    b = TeacherDecision.from_json(payload(), teacher_id="t2", revision="r2", prompt_sha256="a" * 64)
    result = consensus_label(a, b, deterministic_reason="SAME_MODEL")
    assert result["admitted"] is True
    assert result["target"] == 1
    assert result["reason_code"] == "SAME_MODEL"
    assert result["teacher_ids"] == ["t1", "t2"]
