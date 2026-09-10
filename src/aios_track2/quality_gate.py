from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    min_r2: float = 0.95
    max_nrmse: float = 0.05
    min_spearman: float = 0.95
    min_pairwise_accuracy: float = 0.95
    min_top_k_recall: float = 0.90
    max_physics_violation_rate: float = 0.0


DEFAULT_QUALITY_THRESHOLDS = QualityThresholds()


@dataclass(frozen=True, slots=True)
class QualityGateReport:
    passed: bool
    minimum_quality_metric: float
    failures: tuple[str, ...]
    dynamic: dict[str, float]
    ranking: dict[str, float]
    physics_violation_rate: float


def evaluate_quality_gate(
    *,
    dynamic: dict[str, float],
    ranking: dict[str, float],
    physics_violation_rate: float,
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
) -> QualityGateReport:
    failures: list[str] = []
    if dynamic["r2"] < thresholds.min_r2:
        failures.append("R2_LT_095")
    if dynamic["nrmse"] > thresholds.max_nrmse:
        failures.append("NRMSE_GT_005")
    if ranking["spearman"] < thresholds.min_spearman:
        failures.append("SPEARMAN_LT_095")
    if ranking["pairwise_accuracy"] < thresholds.min_pairwise_accuracy:
        failures.append("PAIRWISE_LT_095")
    if ranking["top_k_recall"] < thresholds.min_top_k_recall:
        failures.append("TOPK_LT_090")
    if physics_violation_rate > thresholds.max_physics_violation_rate:
        failures.append("PHYSICS_VIOLATION")
    independent_quality = (
        float(dynamic["r2"]),
        float(ranking["spearman"]),
        float(ranking["pairwise_accuracy"]),
    )
    return QualityGateReport(
        passed=not failures,
        minimum_quality_metric=min(independent_quality),
        failures=tuple(failures),
        dynamic=dict(dynamic),
        ranking=dict(ranking),
        physics_violation_rate=float(physics_violation_rate),
    )
