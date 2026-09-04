from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any

from aios_track2.economics import calculate_npv
from aios_track2.opm import FlowRequest, run_flow
from aios_track2.optimization import OptimizationRequest, optimize
from aios_track2.schedule import Schedule, project_schedule, write_schedule_text


class PipelineState(StrEnum):
    INGESTED = "INGESTED"
    DIAGNOSED = "DIAGNOSED"
    PLANNED = "PLANNED"
    PROJECTED = "PROJECTED"
    PREDICTED = "PREDICTED"
    PROMOTED = "PROMOTED"
    SIMULATED = "SIMULATED"
    VALUED = "VALUED"
    EXPLAINED = "EXPLAINED"
    PACKAGED = "PACKAGED"


LEGAL = {
    PipelineState.INGESTED: PipelineState.DIAGNOSED,
    PipelineState.DIAGNOSED: PipelineState.PLANNED,
    PipelineState.PLANNED: PipelineState.PROJECTED,
    PipelineState.PROJECTED: PipelineState.PREDICTED,
    PipelineState.PREDICTED: PipelineState.PROMOTED,
    PipelineState.PROMOTED: PipelineState.SIMULATED,
    PipelineState.SIMULATED: PipelineState.VALUED,
    PipelineState.VALUED: PipelineState.EXPLAINED,
    PipelineState.EXPLAINED: PipelineState.PACKAGED,
}


@dataclass(frozen=True)
class AgentEvent:
    actor: str
    action: str
    input_hashes: tuple[str, ...]
    output_hashes: tuple[str, ...]
    timestamp: str


@dataclass
class AuditTrail:
    events: list[AgentEvent] = field(default_factory=list)
    writers: dict[str, set[str]] = field(default_factory=dict)
    schedule_before_explanation_sha256: str = ""

    def record(self, actor: str, action: str, inputs: str, outputs: str, writes: str | None = None) -> None:
        event = AgentEvent(
            actor=actor,
            action=action,
            input_hashes=(sha256(inputs.encode()).hexdigest(),),
            output_hashes=(sha256(outputs.encode()).hexdigest(),),
            timestamp=datetime.now(UTC).isoformat(),
        )
        self.events.append(event)
        if writes:
            self.writers.setdefault(writes, set()).add(actor)

    def writers_for(self, field_name: str) -> set[str]:
        return self.writers.get(field_name, set())


@dataclass
class PipelineResult:
    state: PipelineState
    schedule: Schedule
    npv_mrub: float
    schedule_sha256: str
    audit: AuditTrail
    explanation: str
    backend: str
    metrics: dict[str, Any]


class Orchestrator:
    def __init__(self) -> None:
        self.state = PipelineState.INGESTED
        self.audit = AuditTrail()

    def _advance(self, target: PipelineState) -> None:
        expected = LEGAL[self.state]
        if target != expected:
            raise ValueError(f"illegal transition {self.state} -> {target}, expected {expected}")
        self.state = target

    def run_fixture(self) -> PipelineResult:
        from datetime import date

        from aios_track2.schedule import Control, WellRole

        schedule = Schedule(
            controls=(
                Control(date=date(2007, 1, 1), well="P1", status="OPEN", role=WellRole.PRODUCER, wlpr=120.0),
                Control(date=date(2007, 1, 1), well="I1", status="OPEN", role=WellRole.INJECTOR, wwir=140.0),
            )
        )
        return self.run_schedule(schedule)

    def run_schedule(self, schedule: Schedule, output_dir=None) -> PipelineResult:
        self.state = PipelineState.INGESTED
        self.audit.record("MonitorAgent", "ingest", schedule.sha256, "ok")
        self._advance(PipelineState.DIAGNOSED)
        self.audit.record("ReservoirDiagnosticAgent", "diagnose", schedule.sha256, "wct,compensation")
        self._advance(PipelineState.PLANNED)
        planned = optimize(
            OptimizationRequest(seed=42, population=8, iterations=3, elites=2, wells=("P1", "I1"))
        )
        self.audit.record("PlanningAgent", "plan", schedule.sha256, planned.best.schedule_sha256)
        self._advance(PipelineState.PROJECTED)
        projected = project_schedule(planned.best.schedule)
        if not projected.accepted:
            raise RuntimeError("constraint guard rejected planned schedule")
        self.audit.record("ConstraintGuard", "project", planned.best.schedule_sha256, projected.projected.sha256)
        self._advance(PipelineState.PREDICTED)
        self.audit.record("SurrogateAgent", "predict", projected.projected.sha256, str(planned.best.mean_npv))
        self._advance(PipelineState.PROMOTED)
        self.audit.record("SimulatorAgent", "promote", projected.projected.sha256, "queued")
        self._advance(PipelineState.SIMULATED)
        import tempfile
        from pathlib import Path

        work = Path(output_dir or tempfile.mkdtemp(prefix="aios-pipe-"))
        flow = run_flow(FlowRequest(deck=work / "CASE.DATA", output_dir=work, schedule=projected.projected, seed=42))
        self.audit.record("SimulatorAgent", "simulate", projected.projected.sha256, flow.status)
        self._advance(PipelineState.VALUED)
        if flow.monthly_path is None:
            raise RuntimeError("simulator produced no monthly file")
        import pandas as pd

        npv = calculate_npv(pd.read_parquet(flow.monthly_path))
        self.audit.record("EconomicsAgent", "value", flow.stdout_sha256, str(npv.npv_mrub), writes="npv_mrub")
        schedule_text = write_schedule_text(projected.projected)
        if sha256(schedule_text.encode("utf-8")).hexdigest() != projected.projected.sha256:
            raise RuntimeError("schedule hash drifted before explanation")
        self.audit.schedule_before_explanation_sha256 = projected.projected.sha256
        self._advance(PipelineState.EXPLAINED)
        explanation = explain(self.audit, float(npv.npv_mrub), projected.projected)
        self.audit.record("ExplanationAgent", "explain", projected.projected.sha256, explanation)
        self._advance(PipelineState.PACKAGED)
        self.audit.record("MonitorAgent", "package", projected.projected.sha256, "wells_schedule.inc")
        return PipelineResult(
            state=self.state,
            schedule=projected.projected,
            npv_mrub=float(npv.npv_mrub),
            schedule_sha256=projected.projected.sha256,
            audit=self.audit,
            explanation=explanation,
            backend=flow.backend,
            metrics={"runtime_seconds": flow.runtime_seconds, "status": flow.status},
        )


def explain(audit: AuditTrail, npv: float, schedule: Schedule) -> str:
    actors = ", ".join(event.actor for event in audit.events)
    wells = sorted({control.well for control in schedule.controls})
    return (
        f"Schedule for wells {', '.join(wells)} yields contract NPV {npv:.3f} mln RUB. "
        f"Agents involved: {actors}. LLM did not modify numerical controls."
    )


def run_pipeline(schedule: Schedule | None = None) -> PipelineResult:
    orchestrator = Orchestrator()
    if schedule is None:
        return orchestrator.run_fixture()
    return orchestrator.run_schedule(schedule)
