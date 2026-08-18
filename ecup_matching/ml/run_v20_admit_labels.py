"""D3/D4: combine two teacher logs, calibrate on human truth, then admit only approved generated strata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .v20_admission import admit_strata
from .v20_policy import V20Policy, policy_sha256
from .v20_teacher import TeacherDecision, consensus_label


def _read_jsonl(path: Path) -> dict[tuple[int, int], TeacherDecision]:
    out: dict[tuple[int, int], TeacherDecision] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            if not rec.get("valid"):
                continue
            d = rec["decision"]
            out[(int(rec["id1"]), int(rec["id2"]))] = TeacherDecision(
                verdict=str(d["verdict"]), reason_code=str(d["reason_code"]),
                same_product_type=bool(d["same_product_type"]), brand_left=str(d["brand_left"]),
                brand_right=str(d["brand_right"]), model_left=str(d["model_left"]),
                model_right=str(d["model_right"]), critical_attributes=dict(d["critical_attributes"]),
                conflicts=tuple(d["conflicts"]), evidence=tuple(d["evidence"]),
                teacher_id=str(d["teacher_id"]), revision=str(d["revision"]),
                prompt_sha256=str(d["prompt_sha256"]), raw_sha256=str(d["raw_sha256"]),
            )
    return out


def _consensus_frame(pairs: pd.DataFrame, first: dict, second: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = []
    review = []
    for row in pairs.itertuples(index=False):
        key = (int(row.id1), int(row.id2))
        result = consensus_label(first.get(key), second.get(key), deterministic_reason=str(getattr(row, "reason_code", "OTHER")))
        base = {
            "id1": key[0], "id2": key[1],
            "category": str(getattr(row, "category", "")),
            "stratum": str(getattr(row, "stratum", "")),
            "reason_code_deterministic": str(getattr(row, "reason_code", "OTHER")),
        }
        if result.get("admitted"):
            accepted.append({**base, **result})
        else:
            review.append({**base, "admitted": False, "review_reason": result.get("reason", "unknown")})
    return pd.DataFrame(accepted), pd.DataFrame(review)


def run_audit(*, pairs_path: Path, teacher1: Path, teacher2: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(pairs_path).reset_index(drop=True)
    first, second = _read_jsonl(teacher1), _read_jsonl(teacher2)
    accepted, review = _consensus_frame(pairs, first, second)
    truth = pairs[["id1", "id2", "target", "category", "stratum", "reason_code"]].copy()
    if len(accepted):
        joined = accepted.merge(truth, on=["id1", "id2", "category", "stratum"], how="inner", suffixes=("", "_truth"))
        audit_rows = pd.DataFrame({
            "stratum": joined["stratum"].astype(str),
            "category": joined["category"].astype(str),
            "reason_code": joined["reason_code"].astype(str),
            "truth": (joined["target"].astype(float) >= 0.5).astype(int),
            "pred": joined["target_x"].astype(int) if "target_x" in joined else joined["target"].astype(int),
        })
    else:
        audit_rows = pd.DataFrame(columns=["stratum", "category", "reason_code", "truth", "pred"])
    policy = V20Policy()
    admission = admit_strata(audit_rows, policy)
    admission.update({
        "version": "v20-admission-policy-v1",
        "policy_sha256": policy_sha256(policy),
        "teacher1_manifest": str(teacher1.with_suffix(".manifest.json")),
        "teacher2_manifest": str(teacher2.with_suffix(".manifest.json")),
        "consensus_rows": int(len(accepted)), "review_rows": int(len(review)),
        "human_truth_rows": int(len(pairs)), "sealed_gold_opened": False,
    })
    accepted.to_parquet(output_dir / "human-consensus.parquet", index=False)
    review.to_parquet(output_dir / "human-active-review.parquet", index=False)
    (output_dir / "admission-policy.json").write_text(json.dumps(admission, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return admission


def _stratum_reliability(entry: dict[str, object], target: int) -> float:
    by_label = dict(entry.get("by_predicted_label") or {})
    rec = by_label.get(str(int(target)))
    if not rec:
        return 0.0
    return float(rec.get("lcb", 0.0))


def run_candidates(*, pairs_path: Path, teacher1: Path, teacher2: Path, admission_path: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(pairs_path).reset_index(drop=True)
    first, second = _read_jsonl(teacher1), _read_jsonl(teacher2)
    accepted, review = _consensus_frame(pairs, first, second)
    policy = json.loads(admission_path.read_text(encoding="utf-8"))
    admitted_map = dict(policy.get("strata") or {})
    rows = []
    rejected_policy = 0
    for row in accepted.itertuples(index=False):
        entry = admitted_map.get(str(row.stratum), {})
        if not bool(entry.get("admitted", False)):
            rejected_policy += 1
            continue
        reliability = _stratum_reliability(entry, int(row.target))
        if reliability <= 0:
            rejected_policy += 1
            continue
        rows.append({
            "id1": int(row.id1), "id2": int(row.id2), "target": int(row.target),
            "category": str(row.category), "stratum": str(row.stratum),
            "reason_code": str(row.reason_code), "admitted": True,
            "stratum_reliability": reliability,
            "label_origin": "two_teacher_human_calibrated",
            "teacher_ids": json.dumps(row.teacher_ids),
            "teacher_revisions": json.dumps(row.teacher_revisions),
            "prompt_sha256": json.dumps(row.prompt_sha256),
        })
    admitted = pd.DataFrame(rows)
    admitted.to_parquet(output_dir / "admitted_labels.parquet", index=False)
    review.to_parquet(output_dir / "active_review.parquet", index=False)
    report = {
        "version": "v20-generated-admission-v1", "candidate_rows": int(len(pairs)),
        "teacher_consensus_rows": int(len(accepted)), "admitted_rows": int(len(admitted)),
        "policy_rejected_rows": int(rejected_policy), "teacher_review_rows": int(len(review)),
        "admission_policy_sha256": policy.get("policy_sha256"), "sealed_gold_opened": False,
    }
    (output_dir / "generated-admission.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("audit", "candidates"):
        q = sub.add_parser(name)
        q.add_argument("--pairs", type=Path, required=True)
        q.add_argument("--teacher1", type=Path, required=True)
        q.add_argument("--teacher2", type=Path, required=True)
        q.add_argument("--output-dir", type=Path, required=True)
        if name == "candidates":
            q.add_argument("--admission-policy", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "audit":
        run_audit(pairs_path=a.pairs, teacher1=a.teacher1, teacher2=a.teacher2, output_dir=a.output_dir)
    else:
        run_candidates(pairs_path=a.pairs, teacher1=a.teacher1, teacher2=a.teacher2,
                       admission_path=a.admission_policy, output_dir=a.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
