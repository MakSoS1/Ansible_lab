import json
from pathlib import Path

import pandas as pd

from ecup_matching.ml.run_v20_admit_labels import run_audit


def _decision(verdict: str, target: int, teacher: str, item_id: int):
    reason = "SAME_MODEL" if target else "MODEL_CONFLICT"
    return {
        "valid": True,
        "id1": item_id * 2,
        "id2": item_id * 2 + 1,
        "decision": {
            "verdict": verdict,
            "reason_code": reason,
            "same_product_type": bool(target),
            "brand_left": "a", "brand_right": "a",
            "model_left": "x", "model_right": "x" if target else "y",
            "critical_attributes": {}, "conflicts": [], "evidence": ["x"],
            "teacher_id": teacher, "revision": "r1", "prompt_sha256": "a" * 64,
            "raw_sha256": (("b" if teacher == "t1" else "c") * 56 + f"{item_id:08x}")[:64],
        },
    }


def test_audit_precision_compares_teacher_target_to_human_target(tmp_path: Path):
    # 100 distinct calibration pairs satisfy min support without producing a
    # duplicate-key merge. Teachers agree MATCH; human truth is NON_MATCH.
    pairs = pd.DataFrame([
        {
            "id1": i * 2, "id2": i * 2 + 1, "target": 0, "category": "x",
            "stratum": "x|SAME_MODEL|hard", "reason_code": "SAME_MODEL",
        }
        for i in range(1, 101)
    ])
    pairs_path = tmp_path / "pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)
    logs = []
    for teacher in ("t1", "t2"):
        path = tmp_path / f"{teacher}.jsonl"
        path.write_text(
            "".join(json.dumps(_decision("MATCH", 1, teacher, i)) + "\n" for i in range(1, 101)),
            encoding="utf-8",
        )
        logs.append(path)
    report = run_audit(
        pairs_path=pairs_path, teacher1=logs[0], teacher2=logs[1], output_dir=tmp_path / "out"
    )
    stratum = report["strata"]["x|SAME_MODEL|hard"]
    assert stratum["by_predicted_label"]["1"]["successes"] == 0
    assert stratum["admitted"] is False
