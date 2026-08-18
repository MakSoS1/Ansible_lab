import json
from pathlib import Path

import pandas as pd

from ecup_matching.ml.run_v20_admit_labels import run_audit


def _decision(verdict: str, target: int, teacher: str):
    reason = "SAME_MODEL" if target else "MODEL_CONFLICT"
    return {
        "valid": True,
        "id1": 1,
        "id2": 2,
        "decision": {
            "verdict": verdict,
            "reason_code": reason,
            "same_product_type": bool(target),
            "brand_left": "a", "brand_right": "a",
            "model_left": "x", "model_right": "x" if target else "y",
            "critical_attributes": {}, "conflicts": [], "evidence": ["x"],
            "teacher_id": teacher, "revision": "r1", "prompt_sha256": "a" * 64,
            "raw_sha256": ("b" if teacher == "t1" else "c") * 64,
        },
    }


def test_audit_precision_compares_teacher_target_to_human_target(tmp_path: Path):
    # Both teachers agree MATCH, but human truth is NON_MATCH. A buggy self-merge
    # would count this as a success. Correct code must record zero successes.
    pairs = pd.DataFrame([{
        "id1": 1, "id2": 2, "target": 0, "category": "x",
        "stratum": "x|SAME_MODEL|hard", "reason_code": "SAME_MODEL",
    }])
    pairs_path = tmp_path / "pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)
    logs = []
    for teacher in ("t1", "t2"):
        path = tmp_path / f"{teacher}.jsonl"
        path.write_text(json.dumps(_decision("MATCH", 1, teacher)) + "\n", encoding="utf-8")
        logs.append(path)
    report = run_audit(pairs_path=pairs_path, teacher1=logs[0], teacher2=logs[1], output_dir=tmp_path / "out")
    # support is intentionally insufficient, but the raw accepted calibration
    # table must still have 0 correct teacher decisions.
    stratum = report["strata"].get("x|SAME_MODEL|hard")
    assert stratum is not None
    if "by_predicted_label" in stratum:
        assert stratum["by_predicted_label"]["1"]["successes"] == 0
