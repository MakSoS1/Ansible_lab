from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    name: str
    opm_npv: float
    gate_passed: bool
    violations: int
    opm_calls: int
    robustness: float
    declared_npv_error: float = 0.0


def _eligible(row: CandidateEvaluation) -> bool:
    return row.gate_passed and row.violations == 0 and abs(row.declared_npv_error) <= 1e-6


def choose_competition_winner(rows: list[CandidateEvaluation]) -> CandidateEvaluation:
    eligible = [r for r in rows if _eligible(r)]
    if not eligible:
        raise ValueError("no candidate passed hard validation gates")
    return sorted(eligible, key=lambda r: (-r.opm_npv, -r.robustness, r.opm_calls, r.name))[0]


def summarize_bakeoff(rows: list[CandidateEvaluation]) -> list[dict[str, object]]:
    ranked = sorted(rows, key=lambda r: (not _eligible(r), -r.opm_npv, -r.robustness, r.opm_calls, r.name))
    return [asdict(r) | {"eligible": _eligible(r)} for r in ranked]
