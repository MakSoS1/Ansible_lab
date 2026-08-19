import json
from pathlib import Path

import pandas as pd

from ecup_matching.ml.run_v20_admit_labels import run_audit, run_candidates


def _decision(verdict: str, target: int, teacher: str, item_id: int, *, category_reason: str | None = None):
    reason = category_reason or ("SAME_MODEL" if target else "MODEL_CONFLICT")
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
    assert report["hierarchical"]["predicted_labels"]["1"]["successes"] == 0


def test_candidates_use_hierarchical_reliability_not_legacy_stratum_gate(tmp_path: Path):
    pairs = pd.DataFrame([
        {
            "id1": 2, "id2": 3, "category": "x",
            "stratum": "x|MODEL_CONFLICT|hard", "reason_code": "MODEL_CONFLICT",
        }
    ])
    pairs_path = tmp_path / "pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)
    logs = []
    for teacher in ("t1", "t2"):
        path = tmp_path / f"{teacher}.jsonl"
        path.write_text(
            json.dumps(_decision("NON_MATCH", 0, teacher, 1, category_reason="MODEL_CONFLICT")) + "\n",
            encoding="utf-8",
        )
        logs.append(path)
    policy = {
        "version": "v20-admission-policy-v2",
        "policy_sha256": "d" * 64,
        "hierarchical": {
            "version": "v20-hierarchical-admission-v2",
            "predicted_labels": {"0": {"pass": True, "lcb": 0.996}},
            "reason_labels": {
                "MODEL_CONFLICT": {
                    "0": {"pass": True, "lcb": 0.997},
                    "1": {"pass": False, "lcb": 0.0},
                }
            },
            "reason_diagnostics": {
                "MODEL_CONFLICT": {"support": 1, "passing_labels": ["0"], "pass_any_label": True}
            },
            "categories": {"x": {"pass": True, "lcb": 0.980}},
            "critical_family": {"pass": True, "lcb": 0.960},
        },
        "strata": {},
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    report = run_candidates(
        pairs_path=pairs_path,
        teacher1=logs[0],
        teacher2=logs[1],
        admission_path=policy_path,
        output_dir=tmp_path / "admitted",
    )
    assert report["admitted_rows"] == 1
    admitted = pd.read_parquet(tmp_path / "admitted" / "admitted_labels.parquet")
    assert admitted.iloc[0]["target"] == 0
    assert admitted.iloc[0]["stratum_reliability"] == 0.960
    assert admitted.iloc[0]["label_origin"] == "two_teacher_hierarchical_calibrated_v2"
