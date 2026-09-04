from __future__ import annotations

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
