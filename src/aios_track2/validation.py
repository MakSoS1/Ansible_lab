from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]


DEFAULT_THRESHOLDS = {
    "spearman": (">=", 0.75),
    "top_k_recall": (">=", 0.60),
    "nrmse": ("<=", 0.15),
    "coverage_90": (">=", 0.80),
    "constraint_violation_rate": ("<=", 0.0),
    "opm_npv_regret": ("<=", 0.10),
}


def quality_gate(metrics: dict[str, float]) -> GateResult:
    failures: list[str] = []
    for key, (op, threshold) in DEFAULT_THRESHOLDS.items():
        if key not in metrics:
            failures.append(f"missing:{key}")
            continue
        value = float(metrics[key])
        if op == ">=" and value < threshold:
            failures.append(f"{key}<{threshold}")
        elif op == "<=" and value > threshold:
            failures.append(f"{key}>{threshold}")
    return GateResult(not failures, tuple(failures))
