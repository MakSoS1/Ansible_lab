from __future__ import annotations

from typing import Iterable

import pandas as pd

from .v20_admission import CRITICAL_REASONS, wilson_lower_bound


TEACHER_JSON_VALID_MIN = 0.98
TEACHER_COVERAGE_MIN = 0.70
TEACHER_POSITIVE_LCB_MIN = 0.94
TEACHER_NEGATIVE_LCB_MIN = 0.97
TEACHER_CRITICAL_LCB_MIN = 0.95
TEACHER_VRAM_MAX_GIB = 7.75


def _precision_record(pred: pd.Series, truth: pd.Series) -> dict[str, object]:
    trials = int(len(pred))
    successes = int((pred.astype(int).to_numpy() == truth.astype(int).to_numpy()).sum()) if trials else 0
    lcb = wilson_lower_bound(successes, trials)
    return {
        "trials": trials,
        "successes": successes,
        "precision": float(successes / trials) if trials else None,
        "lcb": float(lcb),
    }


def _macro_accuracy(frame: pd.DataFrame, key: str) -> float | None:
    if frame.empty or key not in frame.columns:
        return None
    scores: list[float] = []
    for _, group in frame.groupby(frame[key].astype(str), sort=True):
        if len(group):
            scores.append(float((group["pred"].astype(int) == group["target"].astype(int)).mean()))
    return float(sum(scores) / len(scores)) if scores else None


def _normalise_labels(labels: pd.DataFrame) -> pd.DataFrame:
    required = {"id1", "id2", "pred"}
    if not required.issubset(labels.columns):
        raise ValueError(f"teacher labels missing columns: {sorted(required - set(labels.columns))}")
    work = labels.copy()
    if "valid" not in work.columns:
        work["valid"] = True
    if "uncertain" not in work.columns:
        work["uncertain"] = False
    if "reason_code" not in work.columns:
        work["reason_code"] = "OTHER"
    work = work.drop_duplicates(["id1", "id2"], keep="first").reset_index(drop=True)
    return work


def score_teacher(
    audit_truth: pd.DataFrame,
    normalized_labels: pd.DataFrame,
    runtime_manifest: dict[str, object],
) -> dict[str, object]:
    required_truth = {"id1", "id2", "target", "reason_code", "category"}
    if not required_truth.issubset(audit_truth.columns):
        raise ValueError(f"audit truth missing columns: {sorted(required_truth - set(audit_truth.columns))}")
    labels = _normalise_labels(normalized_labels)
    truth = audit_truth[list(required_truth)].drop_duplicates(["id1", "id2"], keep="first")
    joined = truth.merge(labels, on=["id1", "id2"], how="left", suffixes=("_truth", "_teacher"))

    valid = joined["valid"].fillna(False).astype(bool)
    pred_numeric = pd.to_numeric(joined["pred"], errors="coerce")
    uncertain = joined["uncertain"].fillna(True).astype(bool)
    covered = valid & (~uncertain) & pred_numeric.isin([0, 1])
    joined["pred"] = pred_numeric
    evaluated = joined.loc[covered].copy()

    positive = evaluated.loc[evaluated["pred"].astype(int) == 1]
    negative = evaluated.loc[evaluated["pred"].astype(int) == 0]
    critical = evaluated.loc[evaluated["reason_code_truth"].astype(str).isin(CRITICAL_REASONS)]

    positive_rec = _precision_record(positive["pred"], positive["target"])
    negative_rec = _precision_record(negative["pred"], negative["target"])
    critical_rec = _precision_record(critical["pred"], critical["target"])

    if len(evaluated):
        teacher_reason = evaluated["reason_code_teacher"].fillna("").astype(str)
        truth_reason = evaluated["reason_code_truth"].fillna("").astype(str)
        reason_agreement = float((teacher_reason == truth_reason).mean())
    else:
        reason_agreement = 0.0

    total = int(len(joined))
    json_valid_rate = float(valid.sum() / total) if total else 0.0
    coverage = float(covered.sum() / total) if total else 0.0
    peak_vram = float(runtime_manifest.get("peak_vram_gib", float("inf")))
    rows_per_second = float(runtime_manifest.get("rows_per_second", 0.0))
    revision = str(runtime_manifest.get("revision") or runtime_manifest.get("resolved_revision") or "")

    failed: list[str] = []
    if json_valid_rate + 1e-15 < TEACHER_JSON_VALID_MIN:
        failed.append("json_valid_rate")
    if coverage + 1e-15 < TEACHER_COVERAGE_MIN:
        failed.append("coverage")
    if float(positive_rec["lcb"]) + 1e-15 < TEACHER_POSITIVE_LCB_MIN:
        failed.append("positive_precision_lcb")
    if float(negative_rec["lcb"]) + 1e-15 < TEACHER_NEGATIVE_LCB_MIN:
        failed.append("negative_precision_lcb")
    if float(critical_rec["lcb"]) + 1e-15 < TEACHER_CRITICAL_LCB_MIN:
        failed.append("critical_precision_lcb")
    if peak_vram > TEACHER_VRAM_MAX_GIB + 1e-12:
        failed.append("peak_vram")

    return {
        "version": "v20-teacher-score-v1",
        "model_id": str(runtime_manifest.get("model_id", "")),
        "revision": revision,
        "family": str(runtime_manifest.get("family", "")),
        "backend": str(runtime_manifest.get("backend", "")),
        "quantization": str(runtime_manifest.get("quantization", "")),
        "json_valid_rate": json_valid_rate,
        "coverage": coverage,
        "positive": positive_rec,
        "negative": negative_rec,
        "critical": critical_rec,
        "reason_checker_agreement": reason_agreement,
        "category_macro_accuracy": _macro_accuracy(evaluated, "category"),
        "reason_macro_accuracy": _macro_accuracy(evaluated.rename(columns={"reason_code_truth": "reason_group"}), "reason_group"),
        "rows_per_second": rows_per_second,
        "peak_vram_gib": peak_vram,
        "failed_gates": failed,
        "eligible": not failed,
        "audit_rows": total,
        "covered_rows": int(covered.sum()),
    }


def _pair_labels(labels: pd.DataFrame, suffix: str) -> pd.DataFrame:
    work = _normalise_labels(labels)
    cols = ["id1", "id2", "pred", "valid", "uncertain", "reason_code"]
    work = work[cols].rename(
        columns={
            "pred": f"pred_{suffix}",
            "valid": f"valid_{suffix}",
            "uncertain": f"uncertain_{suffix}",
            "reason_code": f"reason_{suffix}",
        }
    )
    return work


def score_pair(
    audit_truth: pd.DataFrame,
    first_labels: pd.DataFrame,
    second_labels: pd.DataFrame,
    first_report: dict[str, object],
    second_report: dict[str, object],
) -> dict[str, object]:
    required_truth = {"id1", "id2", "target", "reason_code", "category"}
    if not required_truth.issubset(audit_truth.columns):
        raise ValueError(f"audit truth missing columns: {sorted(required_truth - set(audit_truth.columns))}")
    joined = audit_truth[list(required_truth)].drop_duplicates(["id1", "id2"], keep="first")
    joined = joined.merge(_pair_labels(first_labels, "a"), on=["id1", "id2"], how="left")
    joined = joined.merge(_pair_labels(second_labels, "b"), on=["id1", "id2"], how="left")

    pred_a = pd.to_numeric(joined["pred_a"], errors="coerce")
    pred_b = pd.to_numeric(joined["pred_b"], errors="coerce")
    valid = (
        joined["valid_a"].fillna(False).astype(bool)
        & joined["valid_b"].fillna(False).astype(bool)
        & (~joined["uncertain_a"].fillna(True).astype(bool))
        & (~joined["uncertain_b"].fillna(True).astype(bool))
        & pred_a.isin([0, 1])
        & pred_b.isin([0, 1])
    )
    reason_a = joined["reason_a"].fillna("").astype(str)
    reason_b = joined["reason_b"].fillna("").astype(str)
    deterministic_reason = joined["reason_code"].fillna("").astype(str)
    reason_agreement = reason_a == reason_b
    checker_compatible = deterministic_reason.isin(["OTHER", "SPARSE_EVIDENCE"]) | (reason_a == deterministic_reason)
    consensus = valid & (pred_a == pred_b) & reason_agreement & checker_compatible
    got = joined.loc[consensus].copy()
    got["pred"] = pred_a.loc[consensus].astype(int)

    overall = _precision_record(got["pred"], got["target"])
    critical = got.loc[got["reason_code"].astype(str).isin(CRITICAL_REASONS)]
    critical_rec = _precision_record(critical["pred"], critical["target"])
    coverage = float(len(got) / len(joined)) if len(joined) else 0.0
    throughput = min(float(first_report.get("rows_per_second", 0.0)), float(second_report.get("rows_per_second", 0.0)))

    failed: list[str] = []
    if not bool(first_report.get("eligible")) or not bool(second_report.get("eligible")):
        failed.append("teacher_eligibility")
    if str(first_report.get("family", "")) == str(second_report.get("family", "")):
        failed.append("independent_family")
    if (
        str(first_report.get("model_id", "")) == str(second_report.get("model_id", ""))
        and str(first_report.get("revision", "")) == str(second_report.get("revision", ""))
    ):
        failed.append("independent_revision")
    if len(got) == 0:
        failed.append("consensus_coverage")

    return {
        "version": "v20-teacher-pair-score-v1",
        "teacher_models": [str(first_report.get("model_id", "")), str(second_report.get("model_id", ""))],
        "teacher_revisions": [str(first_report.get("revision", "")), str(second_report.get("revision", ""))],
        "teacher_families": [str(first_report.get("family", "")), str(second_report.get("family", ""))],
        "consensus_precision": float(overall["precision"] or 0.0),
        "consensus_lcb": float(overall["lcb"]),
        "critical_precision": float(critical_rec["precision"] or 0.0),
        "critical_lcb": float(critical_rec["lcb"]),
        "coverage": coverage,
        "rows_per_second": throughput,
        "consensus_rows": int(len(got)),
        "failed_gates": failed,
        "eligible": not failed,
    }


def select_teacher_pair(
    teacher_reports: dict[str, dict[str, object]],
    pair_reports: Iterable[dict[str, object]],
) -> dict[str, object]:
    eligible: list[dict[str, object]] = []
    for pair in pair_reports:
        teachers = list(pair.get("teachers") or [])
        if len(teachers) != 2 or not bool(pair.get("eligible")):
            continue
        first = teacher_reports.get(str(teachers[0]))
        second = teacher_reports.get(str(teachers[1]))
        if first is None or second is None:
            continue
        if not bool(first.get("eligible")) or not bool(second.get("eligible")):
            continue
        if str(first.get("family", "")) == str(second.get("family", "")):
            continue
        eligible.append(dict(pair))
    if not eligible:
        raise RuntimeError("no eligible teacher pair")

    def rank(pair: dict[str, object]) -> tuple[float, float, float, float, str]:
        return (
            float(pair.get("consensus_precision", 0.0)),
            float(pair.get("critical_precision", 0.0)),
            float(pair.get("coverage", 0.0)),
            float(pair.get("rows_per_second", 0.0)),
            "|".join(map(str, pair.get("teachers") or [])),
        )

    best = max(eligible, key=rank)
    return {
        "version": "v20-teacher-selection-v1",
        "selected": list(best["teachers"]),
        "best_pair": best,
        "eligible_pairs": int(len(eligible)),
    }


__all__ = [
    "TEACHER_JSON_VALID_MIN",
    "TEACHER_COVERAGE_MIN",
    "TEACHER_POSITIVE_LCB_MIN",
    "TEACHER_NEGATIVE_LCB_MIN",
    "TEACHER_CRITICAL_LCB_MIN",
    "TEACHER_VRAM_MAX_GIB",
    "score_teacher",
    "score_pair",
    "select_teacher_pair",
]
