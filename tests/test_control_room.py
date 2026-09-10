from pathlib import Path

from aios_track2.control_room import EXPECTED_CLEAN_NPV_MRUB, EXPECTED_SCHEDULE_SHA256, load_control_room
from aios_track2.quality_gate import evaluate_quality_gate


def test_control_room_reads_verified_submission_without_demo_names() -> None:
    room = load_control_room(Path("submission"))
    rec = room["recommendation"]
    assert rec["name"] == "mappo"
    assert rec["npv_mrub"] == EXPECTED_CLEAN_NPV_MRUB
    assert rec["schedule_sha256"] == EXPECTED_SCHEDULE_SHA256
    assert rec["sha_matches_clean_rerun"] is True
    assert rec["well_count"] == 103
    assert rec["policy"]["dimensions"] == 18
    assert len(room["wells"]) == 103
    assert all("A-17" not in well["name"] and "John" not in well["name"] for well in room["wells"])
    assert [row["name"] for row in room["compare"]] == ["baseline", "cma_es", "mappo"]
    assert room["holdout"]["preregistered_gate_passed"] is False
    failures = room["holdout"]["preregistered_failures"]
    assert "TOPK_LT_090" in failures or "NPV_TOP3_RECALL_LT_090" in failures


def test_preregistered_gate_on_room_holdout_still_fails_topk() -> None:
    room = load_control_room(Path("submission"))
    holdout = room["holdout"]
    report = evaluate_quality_gate(
        dynamic={"r2": holdout["min_dynamic_r2"], "nrmse": holdout["max_dynamic_nrmse"]},
        ranking={
            "spearman": holdout["spearman"],
            "pairwise_accuracy": holdout["pairwise_accuracy"],
            "top_k_recall": holdout["top_k_recall"],
        },
        physics_violation_rate=0.0,
    )
    assert report.passed is False
    assert "TOPK_LT_090" in report.failures
