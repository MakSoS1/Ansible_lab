from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any


def _telemetry_change_is_safe(changed_files: object) -> bool:
    if not isinstance(changed_files, (list, tuple)) or len(changed_files) != 1:
        return False
    return PurePosixPath(str(changed_files[0])).name == "Model_Z_summary.inc"


def physical_inventory_gate(
    manifests: Mapping[int, Mapping[str, Any]],
    *,
    expected_ids: set[int],
    max_wlpr: float = 500.0,
) -> dict[str, Any]:
    ids = set(int(value) for value in manifests)
    missing = sorted(expected_ids - ids)
    extra = sorted(ids - expected_ids)
    failed_status = sorted(
        scenario_id for scenario_id in expected_ids & ids if manifests[scenario_id].get("status") != "success"
    )
    wlpr_violations = sorted(
        scenario_id
        for scenario_id in expected_ids & ids
        if float(manifests[scenario_id].get("compact_summary", {}).get("max_wlpr", float("inf"))) > max_wlpr + 1e-4
    )
    telemetry_mutations = sorted(
        scenario_id
        for scenario_id in expected_ids & ids
        if not _telemetry_change_is_safe(manifests[scenario_id].get("summary_install_changed_files"))
    )
    baseline_identical = bool(manifests.get(0, {}).get("baseline_schedule_byte_identical"))
    observed_max = max(
        (
            float(manifests[scenario_id].get("compact_summary", {}).get("max_wlpr", float("inf")))
            for scenario_id in expected_ids & ids
        ),
        default=float("inf"),
    )
    passed = not (missing or extra or failed_status or wlpr_violations or telemetry_mutations) and baseline_identical
    return {
        "passed": bool(passed),
        "missing_ids": missing,
        "extra_ids": extra,
        "failed_status_ids": failed_status,
        "wlpr_violations": wlpr_violations,
        "unexpected_telemetry_mutations": telemetry_mutations,
        "baseline_schedule_byte_identical": baseline_identical,
        "max_wlpr": float(observed_max),
    }


def real_opm_tournament_gate(
    evaluation: Mapping[str, Any],
    *,
    min_dynamic_r2: float = 0.95,
    max_dynamic_nrmse: float = 0.05,
    min_spearman: float = 0.95,
    min_pairwise_accuracy: float = 0.95,
    max_simple_regret_mrub: float = 1e-9,
) -> dict[str, Any]:
    """Authorize a real-OPM tournament without rewriting the surrogate audit.

    The preregistered holdout result remains untouched in ``evaluation``.  This
    gate only decides whether it is defensible to spend additional simulator
    calls on finalists whose final ranking will be determined by real OPM NPV,
    not by the surrogate.  Top-k set recall is therefore retained as an audited
    diagnostic but is not used as an authorization criterion; the selected-best
    regret, global rank correlation, and pairwise ordering are required instead.
    """

    physical = evaluation.get("physical_gate", {})
    parity = evaluation.get("reference_parity", {})
    dynamic = evaluation.get("dynamic_selection", {}).get("holdout", {})
    npv = evaluation.get("npv_selection", {}).get("holdout", {})

    failures: list[str] = []
    if not bool(physical.get("passed")):
        failures.append("PHYSICAL_GATE_FAILED")
    if not bool(parity.get("passed")):
        failures.append("REFERENCE_PARITY_FAILED")

    dynamic_r2 = float(dynamic.get("min_aggregate_channel_r2", float("-inf")))
    dynamic_nrmse = float(dynamic.get("max_aggregate_channel_nrmse", float("inf")))
    spearman = float(npv.get("spearman", float("nan")))
    pairwise = float(npv.get("pairwise_accuracy", float("-inf")))
    simple_regret = float(npv.get("simple_regret", float("inf")))
    top_k_recall = float(npv.get("top_k_recall", float("nan")))

    if not math.isfinite(dynamic_r2) or dynamic_r2 < min_dynamic_r2:
        failures.append("DYNAMIC_R2_LT_095")
    if not math.isfinite(dynamic_nrmse) or dynamic_nrmse > max_dynamic_nrmse:
        failures.append("DYNAMIC_NRMSE_GT_005")
    if not math.isfinite(spearman) or spearman < min_spearman:
        failures.append("NPV_SPEARMAN_LT_095")
    if not math.isfinite(pairwise) or pairwise < min_pairwise_accuracy:
        failures.append("NPV_PAIRWISE_LT_095")
    if not math.isfinite(simple_regret) or simple_regret > max_simple_regret_mrub:
        failures.append("NPV_SIMPLE_REGRET_GT_0")

    return {
        "passed": not failures,
        "failures": failures,
        "surrogate_holdout_passed": bool(evaluation.get("passed")),
        "surrogate_holdout_failures": list(evaluation.get("failures", [])),
        "audited_top_k_recall": top_k_recall,
        "criteria": {
            "min_dynamic_r2": float(min_dynamic_r2),
            "max_dynamic_nrmse": float(max_dynamic_nrmse),
            "min_spearman": float(min_spearman),
            "min_pairwise_accuracy": float(min_pairwise_accuracy),
            "max_simple_regret_mrub": float(max_simple_regret_mrub),
        },
        "observed": {
            "min_dynamic_r2": dynamic_r2,
            "max_dynamic_nrmse": dynamic_nrmse,
            "spearman": spearman,
            "pairwise_accuracy": pairwise,
            "simple_regret_mrub": simple_regret,
            "top_k_recall": top_k_recall,
        },
    }
