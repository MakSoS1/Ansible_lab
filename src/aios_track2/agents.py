from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class DecisionState:
    candidate_id: str
    surrogate_npv: float
    uncertainty: float
    status: str = "new"
    constraints_accepted: bool = False
    opm_npv: float | None = None
    audit: tuple[str, ...] = ()

    @property
    def publishable(self) -> bool:
        return self.constraints_accepted and self.opm_npv is not None and self.status == "opm_validated"


class AgentOrchestrator:
    def plan(self, state: DecisionState) -> DecisionState:
        return replace(state, status="planned", audit=state.audit + ("PlanningAgent: candidate prepared",))

    def guard(self, state: DecisionState, *, accepted: bool) -> DecisionState:
        status = "guarded" if accepted else "rejected"
        return replace(state, status=status, constraints_accepted=accepted,
                       audit=state.audit + (f"ConstraintGuard: accepted={accepted}",))

    def record_opm(self, state: DecisionState, *, opm_npv: float | None) -> DecisionState:
        if not state.constraints_accepted:
            return replace(state, status="rejected", audit=state.audit + ("SimulatorAgent: blocked by guard",))
        if opm_npv is None:
            return replace(state, status="needs_opm", audit=state.audit + ("SimulatorAgent: OPM required",))
        return replace(state, status="opm_validated", opm_npv=float(opm_npv),
                       audit=state.audit + (f"EconomicsAgent: OPM NPV={opm_npv}",))
