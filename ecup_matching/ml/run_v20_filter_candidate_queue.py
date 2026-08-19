"""Filter target-free generated candidates using fold-safe pre-inference hierarchical gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .v20_admission import CRITICAL_REASONS


def _hierarchical(policy: dict[str, object]) -> dict[str, object]:
    report = dict(policy.get("hierarchical") or {})
    if report.get("version") != "v20-hierarchical-admission-v1":
        raise ValueError("candidate queue requires v20.1 hierarchical policy")
    return report


def _preinference_pass(row: object, report: dict[str, object]) -> bool:
    reason = str(getattr(row, "reason_code"))
    category = str(getattr(row, "category"))
    reasons = dict(report.get("reasons") or {})
    categories = dict(report.get("categories") or {})
    if not bool(dict(reasons.get(reason) or {}).get("pass", False)):
        return False
    if not bool(dict(categories.get(category) or {}).get("pass", False)):
        return False
    if reason in CRITICAL_REASONS:
        if not bool(dict(report.get("critical_family") or {}).get("pass", False)):
            return False
    return True


def filter_candidate_queue(
    candidates: pd.DataFrame,
    policies: list[dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"id1", "id2", "category", "reason_code", "stratum"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"candidates missing columns: {sorted(required - set(candidates.columns))}")
    if "target" in candidates.columns:
        raise ValueError("candidate teacher queue must remain target-free")
    if len(policies) != 2:
        raise ValueError("candidate queue requires exactly two fold policies")
    reports = [_hierarchical(dict(policy)) for policy in policies]

    keep: list[bool] = []
    rejected_by_fold = [0, 0]
    for row in candidates.itertuples(index=False):
        passed = []
        for index, report in enumerate(reports):
            ok = _preinference_pass(row, report)
            passed.append(ok)
            if not ok:
                rejected_by_fold[index] += 1
        keep.append(all(passed))
    out = candidates.loc[keep].copy().reset_index(drop=True)
    report = {
        "version": "v20-staged-candidate-queue-v1",
        "input_rows": int(len(candidates)),
        "output_rows": int(len(out)),
        "rejected_rows": int(len(candidates) - len(out)),
        "rejected_by_fold": {"0": int(rejected_by_fold[0]), "1": int(rejected_by_fold[1])},
        "target_column_present": bool("target" in out.columns),
        "sealed_gold_opened": False,
    }
    return out, report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--policy", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if len(a.policy) != 2:
        p.error("--policy must be supplied exactly twice")
    candidates = pd.read_parquet(a.candidates)
    policies = [json.loads(path.read_text(encoding="utf-8")) for path in a.policy]
    out, report = filter_candidate_queue(candidates, policies)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.output, index=False)
    a.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
