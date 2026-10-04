from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class VerifiedCandidate:
    name: str
    vector: tuple[float, ...]
    opm_npv_mrub: float
    max_wlpr: float
    status: str
    schedule_sha256: str
    robustness_npvs_mrub: tuple[float, ...] = ()
    opm_calls: int = 1

    @property
    def robustness_floor_mrub(self) -> float:
        values = self.robustness_npvs_mrub or (self.opm_npv_mrub,)
        return float(min(values))

    @property
    def robustness_mean_mrub(self) -> float:
        values = self.robustness_npvs_mrub or (self.opm_npv_mrub,)
        return float(np.mean(values))

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["robustness_floor_mrub"] = self.robustness_floor_mrub
        payload["robustness_mean_mrub"] = self.robustness_mean_mrub
        return payload


def candidate_is_eligible(candidate: VerifiedCandidate, *, max_wlpr: float = 500.0) -> bool:
    return (
        candidate.status == "success"
        and np.isfinite(candidate.opm_npv_mrub)
        and candidate.max_wlpr <= max_wlpr + 1e-4
        and bool(candidate.schedule_sha256)
        and all(np.isfinite(value) for value in candidate.robustness_npvs_mrub)
    )


def choose_verified_winner(
    candidates: Iterable[VerifiedCandidate],
    *,
    max_wlpr: float = 500.0,
) -> VerifiedCandidate:
    eligible = [candidate for candidate in candidates if candidate_is_eligible(candidate, max_wlpr=max_wlpr)]
    if not eligible:
        raise ValueError("no real-OPM candidate passed the hard competition gates")
    return sorted(
        eligible,
        key=lambda candidate: (
            -candidate.opm_npv_mrub,
            -candidate.robustness_floor_mrub,
            -candidate.robustness_mean_mrub,
            candidate.opm_calls,
            candidate.name,
        ),
    )[0]


def verify_clean_rerun(
    winner: VerifiedCandidate,
    *,
    clean_status: str,
    clean_schedule_sha256: str,
    clean_npv_mrub: float,
    clean_max_wlpr: float,
    npv_abs_tolerance_mrub: float = 1e-6,
    max_wlpr: float = 500.0,
) -> dict[str, Any]:
    failures: list[str] = []
    if clean_status != "success":
        failures.append("CLEAN_OPM_FAILED")
    if clean_schedule_sha256 != winner.schedule_sha256:
        failures.append("SCHEDULE_SHA_MISMATCH")
    if not np.isfinite(clean_npv_mrub) or abs(clean_npv_mrub - winner.opm_npv_mrub) > npv_abs_tolerance_mrub:
        failures.append("NPV_MISMATCH")
    if clean_max_wlpr > max_wlpr + 1e-4:
        failures.append("WLPR_GT_500")
    return {
        "passed": not failures,
        "failures": failures,
        "winner": winner.as_dict(),
        "clean_status": clean_status,
        "clean_schedule_sha256": clean_schedule_sha256,
        "clean_npv_mrub": float(clean_npv_mrub),
        "clean_max_wlpr": float(clean_max_wlpr),
        "npv_abs_error_mrub": float(abs(clean_npv_mrub - winner.opm_npv_mrub)),
    }
