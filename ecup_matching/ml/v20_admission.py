from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np
import pandas as pd

from .v20_policy import V20Policy


CRITICAL_REASONS = {
    "MODEL_CONFLICT", "CAPACITY_CONFLICT", "SIZE_CONFLICT", "PACK_COUNT_CONFLICT",
    "ACCESSORY", "DIFFERENT_GENERATION", "BRAND_CONFLICT",
}


def wilson_lower_bound(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    n = int(trials)
    s = int(successes)
    if n <= 0:
        return 0.0
    if s < 0 or s > n:
        raise ValueError("successes must be inside [0,trials]")
    p = s / n
    z2 = float(z) ** 2
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    radius = float(z) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return float(max(0.0, (centre - radius) / denom))


def _stable_component_key(root: object, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{root}".encode("utf-8")).digest()


def _component_roots(frame: pd.DataFrame) -> dict[object, object]:
    parent: dict[object, object] = {}

    def find(x: object) -> object:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    for a, b in frame[["id1", "id2"]].itertuples(index=False, name=None):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {node: find(node) for node in list(parent)}


def build_fold_safe_audit_split(
    frame: pd.DataFrame,
    *,
    audit_fraction: float = 0.10,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not 0.0 < float(audit_fraction) < 0.5:
        raise ValueError("audit_fraction must be inside (0,0.5)")
    required = {"id1", "id2", "target", "category", "stratum"}
    if not required.issubset(frame.columns):
        raise ValueError(f"audit frame missing columns: {sorted(required - set(frame.columns))}")
    if frame.empty:
        raise ValueError("audit frame must not be empty")

    roots = _component_roots(frame)
    row_roots = frame["id1"].map(roots)
    work = frame.copy().reset_index(drop=True)
    work["_root"] = row_roots.to_numpy()
    work["_audit_key"] = (
        work["category"].astype(str) + "\x1f" + work["target"].astype(int).astype(str)
        + "\x1f" + work["stratum"].astype(str)
    )
    total_by_key = work["_audit_key"].value_counts().to_dict()
    desired = {k: max(1.0, float(v) * float(audit_fraction)) for k, v in total_by_key.items()}
    current = {k: 0 for k in total_by_key}
    root_counts = {
        root: group["_audit_key"].value_counts().to_dict()
        for root, group in work.groupby("_root", sort=False)
    }
    target_rows = max(1, int(round(len(work) * float(audit_fraction))))
    selected: set[object] = set()
    selected_rows = 0
    ordered = sorted(root_counts, key=lambda r: _stable_component_key(r, seed))
    for root in ordered:
        if selected_rows >= target_rows:
            break
        counts = root_counts[root]
        gain = sum(min(float(count), max(0.0, desired[k] - current[k])) for k, count in counts.items())
        if gain <= 0 and selected_rows >= target_rows // 2:
            continue
        selected.add(root)
        selected_rows += int(sum(counts.values()))
        for key, count in counts.items():
            current[key] += int(count)
    if not selected and ordered:
        selected.add(ordered[0])

    mask = work["_root"].isin(selected)
    audit = work.loc[mask].drop(columns=["_root", "_audit_key"]).reset_index(drop=True)
    train = work.loc[~mask].drop(columns=["_root", "_audit_key"]).reset_index(drop=True)
    if train.empty or audit.empty:
        raise RuntimeError("component audit split produced an empty side")
    train_items = set(train["id1"]) | set(train["id2"])
    audit_items = set(audit["id1"]) | set(audit["id2"])
    overlap = train_items & audit_items
    if overlap:
        raise RuntimeError(f"component audit split leaked {len(overlap)} items")
    report = {
        "train_rows": int(len(train)),
        "audit_rows": int(len(audit)),
        "realised_fraction": float(len(audit) / len(work)),
        "item_overlap": 0,
        "audit_components": int(len(selected)),
        "audit_strata": int(audit["stratum"].astype(str).nunique()),
    }
    return train, audit, report


def _precision_record(group: pd.DataFrame, *, floor: float) -> dict[str, object]:
    trials = int(len(group))
    successes = int((group["truth"].astype(int) == group["pred"].astype(int)).sum())
    lcb = wilson_lower_bound(successes, trials)
    return {
        "trials": trials,
        "successes": successes,
        "precision": float(successes / trials) if trials else None,
        "lcb": float(lcb),
        "floor": float(floor),
        "pass": bool(trials > 0 and lcb + 1e-15 >= float(floor)),
    }


def _supported_precision_record(
    group: pd.DataFrame,
    *,
    floor: float,
    min_support: int,
) -> dict[str, object]:
    rec = _precision_record(group, floor=floor)
    rec["min_support"] = int(min_support)
    rec["support_pass"] = bool(int(rec["trials"]) >= int(min_support))
    rec["pass"] = bool(rec["pass"] and rec["support_pass"])
    return rec


def build_hierarchical_policy(
    audit_rows: pd.DataFrame,
    policy: V20Policy | None = None,
) -> dict[str, object]:
    """Calibrate teacher consensus at the reliability levels actually reused for candidates.

    Unlike the legacy fine-stratum gate, this pools evidence by predicted label,
    teacher reason, category and the critical-conflict family. Wilson floors are
    unchanged; only support fragmentation is removed.
    """
    policy = policy or V20Policy()
    required = {"truth", "pred", "reason_code", "category"}
    if not required.issubset(audit_rows.columns):
        raise ValueError(f"audit rows missing columns: {sorted(required - set(audit_rows.columns))}")
    work = audit_rows.copy()
    work = work[work["pred"].isin([0, 1])].reset_index(drop=True)
    min_support = int(policy.min_stratum_support)

    labels: dict[str, dict[str, object]] = {}
    for pred_value in (0, 1):
        group = work.loc[work["pred"].astype(int) == pred_value]
        floor = policy.positive_precision_lcb if pred_value == 1 else policy.negative_precision_lcb
        labels[str(pred_value)] = _supported_precision_record(
            group, floor=floor, min_support=min_support
        )

    reasons: dict[str, dict[str, object]] = {}
    for reason, group in work.groupby(work["reason_code"].astype(str), sort=True):
        predicted = sorted(set(group["pred"].astype(int)))
        if len(predicted) != 1:
            rec = _supported_precision_record(
                group, floor=max(policy.positive_precision_lcb, policy.negative_precision_lcb),
                min_support=min_support,
            )
            rec["mixed_predicted_labels"] = True
            rec["pass"] = False
        else:
            floor = policy.positive_precision_lcb if predicted[0] == 1 else policy.negative_precision_lcb
            rec = _supported_precision_record(group, floor=floor, min_support=min_support)
            rec["mixed_predicted_labels"] = False
        reasons[str(reason)] = rec

    categories: dict[str, dict[str, object]] = {}
    for category, group in work.groupby(work["category"].astype(str), sort=True):
        categories[str(category)] = _supported_precision_record(
            group, floor=policy.category_precision_lcb, min_support=min_support
        )

    critical = work.loc[work["reason_code"].astype(str).isin(CRITICAL_REASONS)]
    critical_record = _supported_precision_record(
        critical, floor=policy.critical_precision_lcb, min_support=min_support
    )

    return {
        "version": "v20-hierarchical-admission-v1",
        "policy": policy.__dict__,
        "predicted_labels": labels,
        "reasons": reasons,
        "categories": categories,
        "critical_family": critical_record,
        "audit_rows": int(len(work)),
    }


def row_passes_hierarchical_policy(
    row: object,
    policy_report: dict[str, object],
) -> bool:
    def value(name: str, default=None):
        if isinstance(row, dict):
            return row.get(name, default)
        return getattr(row, name, default)

    try:
        pred = str(int(value("pred", value("target"))))
    except (TypeError, ValueError):
        return False
    reason = str(value("reason_code", ""))
    category = str(value("category", ""))

    labels = dict(policy_report.get("predicted_labels") or {})
    reasons = dict(policy_report.get("reasons") or {})
    categories = dict(policy_report.get("categories") or {})
    if not bool(dict(labels.get(pred) or {}).get("pass", False)):
        return False
    if not bool(dict(reasons.get(reason) or {}).get("pass", False)):
        return False
    if not bool(dict(categories.get(category) or {}).get("pass", False)):
        return False
    if reason in CRITICAL_REASONS:
        if not bool(dict(policy_report.get("critical_family") or {}).get("pass", False)):
            return False
    return True


def admit_strata(audit_rows: pd.DataFrame, policy: V20Policy | None = None) -> dict[str, object]:
    policy = policy or V20Policy()
    required = {"stratum", "truth", "pred", "reason_code"}
    if not required.issubset(audit_rows.columns):
        raise ValueError(f"audit rows missing columns: {sorted(required - set(audit_rows.columns))}")
    work = audit_rows.copy()
    work = work[work["pred"].isin([0, 1])].reset_index(drop=True)

    category_pass: dict[str, bool] = {}
    category_report: dict[str, object] = {}
    if "category" in work.columns:
        for category, group in work.groupby(work["category"].astype(str), sort=True):
            rec = _precision_record(group, floor=policy.category_precision_lcb)
            category_report[str(category)] = rec
            category_pass[str(category)] = bool(rec["pass"])

    strata: dict[str, object] = {}
    admitted: list[str] = []
    for stratum, group in work.groupby(work["stratum"].astype(str), sort=True):
        total = int(len(group))
        if total < int(policy.min_stratum_support):
            strata[str(stratum)] = {
                "admitted": False,
                "reason": "insufficient_support",
                "support": total,
            }
            continue
        label_records: dict[str, object] = {}
        passes = []
        for pred_value, pred_group in group.groupby("pred", sort=True):
            floor = policy.positive_precision_lcb if int(pred_value) == 1 else policy.negative_precision_lcb
            rec = _precision_record(pred_group, floor=floor)
            label_records[str(int(pred_value))] = rec
            passes.append(bool(rec["pass"]))
        reasons = set(group["reason_code"].astype(str))
        critical = bool(reasons & CRITICAL_REASONS)
        critical_record = None
        if critical:
            critical_record = _precision_record(group, floor=policy.critical_precision_lcb)
            passes.append(bool(critical_record["pass"]))
        if "category" in group.columns:
            cats = set(group["category"].astype(str))
            passes.extend(category_pass.get(cat, False) for cat in cats)
        ok = bool(passes and all(passes))
        strata[str(stratum)] = {
            "admitted": ok,
            "reason": "passed" if ok else "precision_below_floor",
            "support": total,
            "by_predicted_label": label_records,
            "critical": critical,
            "critical_precision": critical_record,
        }
        if ok:
            admitted.append(str(stratum))
    return {
        "policy": policy.__dict__,
        "categories": category_report,
        "strata": strata,
        "admitted_strata": admitted,
        "admitted_count": int(len(admitted)),
    }


__all__ = [
    "CRITICAL_REASONS",
    "wilson_lower_bound",
    "build_fold_safe_audit_split",
    "build_hierarchical_policy",
    "row_passes_hierarchical_policy",
    "admit_strata",
]
