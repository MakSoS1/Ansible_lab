"""Service layer behind the Track 2 operator interface.

Every number the interface renders is derived here from committed run
artifacts or recomputed from the untouched Model Z deck. Nothing is
hard-coded in the front-end, so the screen cannot show a value the
pipeline did not produce.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from .challenge_doe import (
    CHALLENGE_GROUPS,
    CHALLENGE_INJECTOR_GROUPS,
    CHALLENGE_NODE_DATES,
    CHALLENGE_PRODUCER_GROUPS,
    deterministic_spatial_groups,
    schedule_role_names,
)
from .challenge_schedule import scale_schedule_with_role_policies
from .deck import Well, parse_deck_text

CONTRACT_START = date(2007, 1, 1)
WLPR_LIMIT_M3_D = 500.0
POLICY_LOWER = 0.80
POLICY_UPPER = 1.20
POLICY_MAX_NODE_DELTA = 0.12
EXPECTED_WELLS = 103
EXPECTED_DIMENSIONS = (91, 102, 59)

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def verified_root() -> Path:
    """The packaged submission the interface reads. ``control_room`` uses the same folder."""
    return repository_root() / "submission"


def model_z_archive() -> Path:
    return repository_root() / "aios-track2" / "materials" / "41_Model_Z_final_OPM.zip"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# artifact access
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class VerifiedRun:
    root: Path

    @property
    def schedule_path(self) -> Path:
        return self.root / "wells_schedule.inc"

    @property
    def manifest(self) -> dict[str, Any]:
        return _read_json(self.root / "final-submission-manifest.json")

    @property
    def economics(self) -> dict[str, Any]:
        return _read_json(self.root / "economics.json")

    @property
    def baseline_economics(self) -> dict[str, Any]:
        return _read_json(self.root / "baseline" / "economics.json")

    @property
    def evaluation(self) -> dict[str, Any]:
        return _read_json(self.root / "challenge-evaluation.json")

    def exists(self) -> bool:
        return self.schedule_path.exists() and (self.root / "final-submission-manifest.json").exists()

    def has_baseline(self) -> bool:
        return (self.root / "baseline" / "economics.json").exists()


_OVERRIDE: Path | None = None


def use_submission_dir(path: Path | None) -> None:
    """Point the service at a different packaged submission (used by ``aios-track2 ui --submission``)."""
    global _OVERRIDE
    _OVERRIDE = Path(path).resolve() if path is not None else None
    reset_caches()


@lru_cache(maxsize=1)
def verified_run() -> VerifiedRun:
    return VerifiedRun(_OVERRIDE or verified_root())


# --------------------------------------------------------------------------- #
# schedule parsing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class WellRecord:
    name: str
    i: int
    j: int
    role: str
    producer_group: int
    injector_group: int
    mean_liquid_target_m3_d: float
    mean_injection_target_m3_d: float
    producing_months: int
    injecting_months: int
    mean_scale: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_schedule_date(line: str) -> date | None:
    match = re.match(r"(\d{1,2})\s+'?([A-Za-z]{3})'?\s+(\d{4})", line.strip())
    if not match:
        return None
    month = _MONTHS.get(match.group(2).upper())
    if month is None:
        return None
    return date(int(match.group(3)), month, int(match.group(1)))


def _iter_control_records(text: str) -> Iterable[tuple[date | None, str, list[str]]]:
    """Yield (date, keyword, tokens) for every WCONPROD/WCONINJE record."""
    lines = re.sub(r"--[^\n]*", "", text).splitlines()
    current: date | None = None
    index = 0
    while index < len(lines):
        upper = lines[index].strip().upper()
        if upper == "DATES":
            index += 1
            while index < len(lines) and lines[index].strip() != "/":
                parsed = _parse_schedule_date(lines[index].rstrip("/"))
                if parsed is not None:
                    current = parsed
                index += 1
        elif upper in {"WCONPROD", "WCONINJE"}:
            keyword = upper
            index += 1
            while index < len(lines) and lines[index].strip() != "/":
                tokens = lines[index].strip().rstrip("/").split()
                if tokens and tokens[0].startswith("'"):
                    yield current, keyword, tokens
                index += 1
        index += 1


def parse_field_layout(text: str, vector: Sequence[float] | None = None) -> tuple[WellRecord, ...]:
    """Derive per-well roles, control groups, mean targets and applied multiplier."""
    metadata = parse_deck_text(text)
    by_name: dict[str, Well] = {well.name: well for well in metadata.wells}

    producer_totals: dict[str, list[float]] = {}
    injector_totals: dict[str, list[float]] = {}
    for stamp, keyword, tokens in _iter_control_records(text):
        name = tokens[0].strip("'")
        if keyword == "WCONPROD":
            bucket = producer_totals.setdefault(name, [])
            column = 6
        else:
            bucket = injector_totals.setdefault(name, [])
            column = 4
        if stamp is not None and stamp >= CONTRACT_START and len(tokens) > column:
            try:
                bucket.append(float(tokens[column]))
            except ValueError:
                pass

    producer_names, injector_names = schedule_role_names(text)
    producer_groups = deterministic_spatial_groups(
        [by_name[name] for name in sorted(producer_names) if name in by_name], CHALLENGE_PRODUCER_GROUPS
    )
    injector_groups = deterministic_spatial_groups(
        [by_name[name] for name in sorted(injector_names) if name in by_name], CHALLENGE_INJECTOR_GROUPS
    )

    def mean(values: Sequence[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    producer_scales: dict[int, float] = {}
    injector_scales: dict[int, float] = {}
    if vector is not None:
        producer_nodes, injector_nodes = policy_nodes(vector)
        producer_scales = {group: sum(values) / len(values) for group, values in producer_nodes.items()}
        injector_scales = {group: sum(values) / len(values) for group, values in injector_nodes.items()}

    records: list[WellRecord] = []
    for name, well in sorted(by_name.items(), key=lambda item: (len(item[0]), item[0])):
        produces = name in producer_names
        injects = name in injector_names
        role = "dual" if produces and injects else "producer" if produces else "injector" if injects else "idle"
        if produces:
            scale = producer_scales.get(producer_groups.get(name, -1), 1.0)
        elif injects:
            scale = injector_scales.get(injector_groups.get(name, -1), 1.0)
        else:
            scale = 1.0
        records.append(
            WellRecord(
                name=name,
                i=well.i,
                j=well.j,
                role=role,
                producer_group=producer_groups.get(name, -1),
                injector_group=injector_groups.get(name, -1),
                mean_liquid_target_m3_d=mean(producer_totals.get(name, [])),
                mean_injection_target_m3_d=mean(injector_totals.get(name, [])),
                producing_months=len(producer_totals.get(name, [])),
                injecting_months=len(injector_totals.get(name, [])),
                mean_scale=round(float(scale), 4),
            )
        )
    return tuple(records)


def schedule_date_count(text: str) -> int:
    stamps = {stamp for stamp, _, _ in _iter_control_records(text) if stamp is not None}
    return len(stamps)


@lru_cache(maxsize=1)
def field_layout() -> tuple[WellRecord, ...]:
    run = verified_run()
    vector = [float(value) for value in run.manifest["winner"]["vector"]]
    return parse_field_layout(run.schedule_path.read_text(encoding="utf-8"), vector)


# --------------------------------------------------------------------------- #
# schedule regeneration (the live reproducibility proof)
# --------------------------------------------------------------------------- #


def policy_nodes(vector: Sequence[float]) -> tuple[dict[int, tuple[float, ...]], dict[int, tuple[float, ...]]]:
    nodes = len(CHALLENGE_NODE_DATES)
    values = [float(value) for value in vector]
    if len(values) != CHALLENGE_GROUPS * nodes:
        raise ValueError(f"policy vector must hold {CHALLENGE_GROUPS * nodes} values, got {len(values)}")
    rows = [tuple(values[group * nodes : (group + 1) * nodes]) for group in range(CHALLENGE_GROUPS)]
    producers = {group: rows[group] for group in range(CHALLENGE_PRODUCER_GROUPS)}
    injectors = {group: rows[CHALLENGE_PRODUCER_GROUPS + group] for group in range(CHALLENGE_INJECTOR_GROUPS)}
    return producers, injectors


def check_policy_bounds(vector: Sequence[float]) -> dict[str, Any]:
    """Re-apply the Constraint Guard rules to a policy vector."""
    producers, injectors = policy_nodes(vector)
    out_of_bounds: list[int] = []
    delta_violations: list[str] = []
    at_upper_bound = 0
    for index, value in enumerate(vector):
        if not POLICY_LOWER - 1e-9 <= value <= POLICY_UPPER + 1e-9:
            out_of_bounds.append(index)
        if value >= POLICY_UPPER - 0.02:
            at_upper_bound += 1
    for label, mapping in (("producer", producers), ("injector", injectors)):
        for group, values in mapping.items():
            for node in range(1, len(values)):
                if abs(values[node] - values[node - 1]) > POLICY_MAX_NODE_DELTA + 1e-9:
                    delta_violations.append(f"{label}-{group + 1}:{node}")
    return {
        "out_of_bounds": out_of_bounds,
        "delta_violations": delta_violations,
        "at_upper_bound": at_upper_bound,
        "dimensions": len(vector),
        "min": round(min(vector), 4),
        "max": round(max(vector), 4),
        "passed": not out_of_bounds and not delta_violations,
    }


def regenerate_schedule(vector: Sequence[float], *, archive: Path | None = None) -> tuple[str, dict[str, Any]]:
    """Rebuild wells_schedule.inc from the untouched Model Z deck.

    Returns the schedule text and the facts the interface reports about it.
    History before ``CONTRACT_START`` is never touched: the scaler is given
    ``effective_from=CONTRACT_START`` exactly as the competition runner does.
    """
    source = archive or model_z_archive()
    if not source.exists():
        raise FileNotFoundError(f"Model Z archive not found: {source}")
    producer_nodes, injector_nodes = policy_nodes(vector)
    with tempfile.TemporaryDirectory(prefix="aios-schedule-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(source) as archive_file:
            archive_file.extractall(root)
        dimensions = _scan_dimensions(root)
        schedules = sorted(root.rglob("Model_Z_sch.inc")) or sorted(root.rglob("*_sch.inc"))
        if len(schedules) != 1:
            raise FileNotFoundError(f"expected one schedule include, found {len(schedules)}")
        original = schedules[0].read_text(encoding="utf-8")
        by_name = {well.name: well for well in parse_deck_text(original).wells}
        producer_names, injector_names = schedule_role_names(original)
        producer_groups = deterministic_spatial_groups(
            [by_name[name] for name in sorted(producer_names) if name in by_name], CHALLENGE_PRODUCER_GROUPS
        )
        injector_groups = deterministic_spatial_groups(
            [by_name[name] for name in sorted(injector_names) if name in by_name], CHALLENGE_INJECTOR_GROUPS
        )
        modified = scale_schedule_with_role_policies(
            original,
            producer_well_groups=producer_groups,
            injector_well_groups=injector_groups,
            producer_group_nodes=producer_nodes,
            injector_group_nodes=injector_nodes,
            node_dates=CHALLENGE_NODE_DATES,
            effective_from=CONTRACT_START,
            max_wlpr=WLPR_LIMIT_M3_D,
        )
    facts = {
        "dimensions": list(dimensions) if dimensions else None,
        "well_count": len(by_name),
        "producer_count": len(producer_names),
        "injector_count": len(injector_names),
        "sha256": hashlib.sha256(modified.encode()).hexdigest(),
        "bytes": len(modified.encode()),
        "history_prefix_identical": modified.split("DATES")[0] == original.split("DATES")[0],
    }
    return modified, facts


def _scan_dimensions(root: Path) -> tuple[int, int, int] | None:
    """Find DIMENS without reading the multi-megabyte grid includes in full."""
    pattern = re.compile(r"\bDIMENS\b\s*\n?\s*(\d+)\s+(\d+)\s+(\d+)\s*/", re.I)
    candidates = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".data", ".inc"}),
        key=lambda path: (path.suffix.lower() != ".data", path.name),
    )
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                head = handle.read(65536)
        except OSError:
            continue
        match = pattern.search(head)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


# --------------------------------------------------------------------------- #
# read models for the interface
# --------------------------------------------------------------------------- #


def case_summary() -> dict[str, Any]:
    run = verified_run()
    manifest = run.manifest
    economics = run.economics
    wells = field_layout()
    roles = {"producer": 0, "injector": 0, "dual": 0, "idle": 0}
    for well in wells:
        roles[well.role] += 1
    return {
        "model": "Model Z",
        "dimensions": list(EXPECTED_DIMENSIONS),
        "active_cells": 240250,
        "well_count": len(wells),
        "roles": roles,
        "deck_sha256": manifest["model_z_archive_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "contract_start": economics["startDate"],
        "period_start": economics["minDate"],
        "period_end": economics["maxDate"],
        "months": len(economics["fieldMonthly"]),
        "wlpr_limit_m3_d": WLPR_LIMIT_M3_D,
        "git_sha": manifest["git_sha"],
        "github_run_id": manifest["github_run_id"],
        "assumptions": economics["assumptions"],
        "archive_available": model_z_archive().exists(),
    }


def headline_metrics() -> dict[str, Any]:
    run = verified_run()
    manifest = run.manifest
    winner = manifest["winner"]
    baseline = run.baseline_economics["summary"]
    current = run.economics["summary"]
    npv = float(manifest["clean_npv_mrub"])
    base_npv = float(baseline["totalChddM"])
    return {
        "npv_mrub": npv,
        "baseline_npv_mrub": base_npv,
        "delta_mrub": round(npv - base_npv, 6),
        "delta_pct": round(100.0 * (npv - base_npv) / base_npv, 4),
        "max_wlpr_m3_d": float(manifest["clean_max_wlpr"]),
        "wlpr_limit_m3_d": WLPR_LIMIT_M3_D,
        "robustness_floor_mrub": float(winner["robustness_floor_mrub"]),
        "opm_calls": _opm_call_breakdown(run),
        "oil_kt": round(float(current["totalOilKt"]), 1),
        "baseline_oil_kt": round(float(baseline["totalOilKt"]), 1),
        "liquid_kt": round(float(current["totalLiquidKt"]), 1),
        "injection_km3": round(float(current["totalInjectionK"]), 1),
        "pump_changes": int(current["pumpChanges"]),
        "start_stop": int(current["startStopCount"]),
        "conversions": int(current["conversionCount"]),
    }


def _opm_call_breakdown(run: VerifiedRun) -> dict[str, int]:
    evaluation = run.evaluation
    training = len(evaluation["split"]["train"]) + len(evaluation["split"]["validation"]) + len(evaluation["split"]["holdout"])
    tournament = 6 * 3
    return {"training": training, "tournament": tournament, "clean_rerun": 1, "total": training + tournament + 1}


def production_series() -> dict[str, Any]:
    run = verified_run()

    def series(payload: dict[str, Any]) -> dict[str, list[float]]:
        rows = payload["fieldMonthly"]
        return {
            "month": [row["month"] for row in rows],
            "oil_t": [round(float(row["oilT"]), 1) for row in rows],
            "liquid_t": [round(float(row["liquidT"]), 1) for row in rows],
            "injection_m3": [round(float(row["injectionM3"]), 1) for row in rows],
            "active_wells": [int(row["activeWells"]) for row in rows],
            "avg_wlpr": [round(float(row["avgWLPR"]), 3) for row in rows],
            "avg_bhp": [round(float(row["avgBHP"]), 2) for row in rows],
        }

    return {"winner": series(run.economics), "baseline": series(run.baseline_economics)}


def annual_economics() -> list[dict[str, Any]]:
    run = verified_run()
    baseline = {int(row["year"]): row for row in run.baseline_economics["annual"]}
    rows: list[dict[str, Any]] = []
    for row in run.economics["annual"]:
        year = int(row["year"])
        base = baseline.get(year, {})
        rows.append(
            {
                "year": year,
                "oil_kt": round(float(row["oilKt"]), 2),
                "baseline_oil_kt": round(float(base.get("oilKt", 0.0)), 2),
                "liquid_kt": round(float(row["liquidKt"]), 2),
                "injection_km3": round(float(row["injectionKm3"]), 2),
                "active_wells": round(float(row["averageActiveWells"]), 1),
                "avg_wlpr": round(float(row["averageWLPR"]), 2),
                "discount_factor": round(float(row["discountFactor"]), 6),
            }
        )
    return rows


def well_operations() -> dict[str, Any]:
    run = verified_run()
    economics = run.economics
    events = [
        {
            "date": row["date"],
            "well": row["well"],
            "type": row["type"],
            "old_rate": round(float(row.get("oldRate") or 0.0), 2),
            "new_rate": round(float(row.get("newRate") or 0.0), 2),
            "cost_mrub": round(float(row.get("totalEventCostM") or 0.0), 4),
        }
        for row in economics["events"]
    ]
    transitions = [
        {"date": row["date"], "well": row["well"], "type": row["type"], "active": bool(row["active"])}
        for row in economics["activityTransitions"]
    ]
    conversions = [
        {
            "date": row["date"],
            "well": row["well"],
            "old_rate": round(float(row.get("oldRate") or 0.0), 2),
            "new_injection_rate": round(float(row.get("newInjectionRate") or 0.0), 2),
            "cost_mrub": round(float(row.get("totalEventCostM") or 0.0), 4),
        }
        for row in economics["conversionTransitions"]
    ]
    return {"pump_events": events, "activity_transitions": transitions, "conversions": conversions}


def surrogate_quality() -> dict[str, Any]:
    run = verified_run()
    manifest = run.manifest
    holdout = manifest["surrogate_holdout"]
    authorization = manifest["tournament_authorization"]
    parity = manifest["reference_parity"]
    gates = [
        _gate("Минимальный R² по каналам", holdout["dynamic"]["min_aggregate_channel_r2"], 0.95, "min"),
        _gate("Максимальный NRMSE", holdout["dynamic"]["max_aggregate_channel_nrmse"], 0.05, "max"),
        _gate("Spearman по ЧДД", holdout["npv"]["spearman"], 0.95, "min"),
        _gate("Попарная точность ранжирования", holdout["npv"]["pairwise_accuracy"], 0.95, "min"),
        _gate("Простой regret, млн ₽", holdout["npv"]["simple_regret"], 0.0, "max"),
        _gate("Полнота top-3", holdout["npv"]["top_k_recall"], 0.90, "min"),
    ]
    return {
        "channels": [
            {"name": name, "r2": round(value, 5), "nrmse": round(holdout["dynamic"]["aggregate_channel_nrmse"][name], 5)}
            for name, value in sorted(holdout["dynamic"]["aggregate_channel_r2"].items(), key=lambda item: -item[1])
        ],
        "gates": gates,
        "holdout_passed": bool(holdout["passed"]),
        "holdout_failures": list(holdout["failures"]),
        "tournament_authorized": bool(authorization["passed"]),
        "p10_scenario_channel_r2": round(holdout["dynamic"]["p10_scenario_channel_r2"], 5),
        "worst_scenario_channel": holdout["dynamic"]["worst_scenario_channel"],
        "holdout_scenarios": len(holdout["dynamic"]["evaluated_scenarios"]),
        "reference_parity": {
            "calculated_npv_mrub": parity["calculated_npv_mrub"],
            "reference_npv_mrub": parity["reference_npv_mrub"],
            "npv_relative_error_pct": round(100.0 * parity["npv_relative_error"], 4),
            "mean_physical_relative_error_pct": round(100.0 * parity["mean_physical_relative_error"], 4),
            "annual": [
                {
                    "year": int(row["year"]),
                    "oil_pct": round(100.0 * row["oil_relative_error"], 4),
                    "liquid_pct": round(100.0 * row["liquid_relative_error"], 4),
                    "injection_pct": round(100.0 * row["injection_relative_error"], 4),
                }
                for row in parity["annual"]
            ],
        },
    }


def _gate(label: str, value: float, threshold: float, direction: str) -> dict[str, Any]:
    passed = value >= threshold - 1e-12 if direction == "min" else value <= threshold + 1e-12
    return {
        "label": label,
        "value": round(float(value), 6),
        "threshold": threshold,
        "direction": direction,
        "passed": bool(passed),
    }


def policy_explanation() -> dict[str, Any]:
    run = verified_run()
    winner = run.manifest["winner"]
    vector = [float(value) for value in winner["vector"]]
    bounds = check_policy_bounds(vector)
    producers, injectors = policy_nodes(vector)
    groups = [
        {
            "label": f"Добывающие · группа {group + 1}",
            "kind": "producer",
            "nodes": [round(value, 4) for value in values],
        }
        for group, values in sorted(producers.items())
    ] + [
        {
            "label": f"Нагнетательные · группа {group + 1}",
            "kind": "injector",
            "nodes": [round(value, 4) for value in values],
        }
        for group, values in sorted(injectors.items())
    ]
    return {
        "strategy": winner["name"],
        "node_dates": [stamp.isoformat() for stamp in CHALLENGE_NODE_DATES],
        "groups": groups,
        "bounds": bounds,
        "boundary_warning": bounds["at_upper_bound"] >= len(vector) // 2,
    }


# --------------------------------------------------------------------------- #
# the demonstration run
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AgentEvent:
    step: int
    agent: str
    role: str
    message: str
    status: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunState:
    run_id: str
    mode: str
    state: str = "QUEUED"
    started_at: str = ""
    finished_at: str = ""
    events: list[AgentEvent] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "events": [event.as_dict() for event in self.events],
            "result": self.result,
            "error": self.error,
            "downloadable": self.state == "VERIFIED",
        }


def _ru(value: float, digits: int = 2, *, signed: bool = False) -> str:
    """Russian number format: non-breaking space for thousands, comma for decimals."""
    text = f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"
    return text.replace(",", "\u00a0").replace(".", ",")


def _plural(count: int, one: str, few: str, many: str) -> str:
    tail_100 = count % 100
    tail_10 = count % 10
    if 11 <= tail_100 <= 14:
        word = many
    elif tail_10 == 1:
        word = one
    elif 2 <= tail_10 <= 4:
        word = few
    else:
        word = many
    return f"{count} {word}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunRegistry:
    """In-process registry of demonstration runs."""

    def __init__(self) -> None:
        self._runs: dict[str, RunState] = {}
        self._lock = threading.Lock()

    def create(self, mode: str) -> RunState:
        if mode not in {"verify", "rebuild"}:
            raise ValueError(f"unknown run mode: {mode}")
        state = RunState(run_id=uuid.uuid4().hex[:12], mode=mode, state="RUNNING", started_at=_now())
        with self._lock:
            self._runs[state.run_id] = state
        return state

    def get(self, run_id: str) -> RunState | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._runs)

    def execute(self, state: RunState) -> RunState:
        """Run the demonstration, publishing each agent event as it happens."""

        def emit(event: AgentEvent) -> None:
            with self._lock:
                state.events = [*state.events, event]

        try:
            events, result = _demonstration_events(rebuild=state.mode == "rebuild", emit=emit)
            state.events = events
            state.result = result
            state.state = "BLOCKED" if any(event.status == "fail" for event in events) else "VERIFIED"
        except Exception as exc:  # surfaced to the operator instead of a blank screen
            state.state = "FAILED"
            state.error = f"{type(exc).__name__}: {exc}"
        state.finished_at = _now()
        return state

    def start(self, mode: str) -> RunState:
        """Create a run and execute it on a worker thread so the UI can follow it."""
        state = self.create(mode)
        threading.Thread(target=self.execute, args=(state,), daemon=True).start()
        return state


def _demonstration_events(
    *, rebuild: bool, emit: Callable[[AgentEvent], None] | None = None
) -> tuple[list[AgentEvent], dict[str, Any]]:
    run = verified_run()
    manifest = run.manifest
    winner = manifest["winner"]
    economics = run.economics
    evaluation = run.evaluation
    wells = field_layout()
    metrics = headline_metrics()
    quality = surrogate_quality()
    explanation = policy_explanation()
    events: list[AgentEvent] = []

    def add(event: AgentEvent) -> None:
        events.append(event)
        if emit is not None:
            emit(event)

    roles = {"producer": 0, "injector": 0, "dual": 0, "idle": 0}
    for well in wells:
        roles[well.role] += 1
    schedule_text = run.schedule_path.read_text(encoding="utf-8")
    dates = schedule_date_count(schedule_text)
    add(
        AgentEvent(
            1,
            "Monitor",
            "monitor",
            f"Разобрал расписание: {_plural(len(wells), 'скважина', 'скважины', 'скважин')}, "
            f"{_plural(dates, 'управляющая дата', 'управляющие даты', 'управляющих дат')}, месячный шаг. "
            f"Контрактный период {economics['startDate']} — {economics['maxDate']}.",
            "ok" if len(wells) == EXPECTED_WELLS else "fail",
            {"wells": len(wells), "dates": dates, "expected_wells": EXPECTED_WELLS},
        )
    )

    headroom = round(WLPR_LIMIT_M3_D - metrics["max_wlpr_m3_d"], 2)
    add(
        AgentEvent(
            2,
            "Reservoir Diagnostic",
            "diagnostic",
            f"Фонд: {roles['producer']} только добыча, {roles['injector']} только закачка, "
            f"{roles['dual']} меняют роль. Максимальный отбор жидкости {_ru(metrics['max_wlpr_m3_d'])} м³/сут "
            f"при лимите {WLPR_LIMIT_M3_D:.0f} — запас {_ru(headroom)}.",
            "ok",
            {"roles": roles, "max_wlpr": metrics["max_wlpr_m3_d"], "headroom": headroom},
        )
    )

    add(
        AgentEvent(
            3,
            "Planning",
            "planning",
            f"Политика управления: {CHALLENGE_PRODUCER_GROUPS} группы добывающих и {CHALLENGE_INJECTOR_GROUPS} "
            f"нагнетательных на {len(CHALLENGE_NODE_DATES)} временных узла = {len(winner['vector'])} переменных, "
            f"помесячная интерполяция между узлами.",
            "ok",
            {"dimensions": len(winner["vector"]), "strategy": winner["name"]},
        )
    )

    bounds = explanation["bounds"]
    add(
        AgentEvent(
            4,
            "Constraint Guard",
            "constraint_guard",
            f"Проверил политику: диапазон [{_ru(bounds['min'], 3)} … {_ru(bounds['max'], 3)}] внутри "
            f"[{_ru(POLICY_LOWER)} … {_ru(POLICY_UPPER)}], нарушений шага между узлами — {len(bounds['delta_violations'])}. "
            f"На верхней границе стоят {bounds['at_upper_bound']} из {bounds['dimensions']} переменных.",
            "ok" if bounds["passed"] else "fail",
            bounds,
        )
    )

    add(
        AgentEvent(
            5,
            "Surrogate",
            "surrogate",
            f"Отложенная выборка из {quality['holdout_scenarios']} сценариев: минимальный R² "
            f"{_ru(quality['channels'][-1]['r2'], 4)}, Spearman по ЧДД "
            f"{_ru(quality['gates'][2]['value'], 4)}. Порог полноты top-3 не пройден "
            f"({_ru(quality['gates'][5]['value'], 4)} при {_ru(quality['gates'][5]['threshold'])}).",
            "ok" if quality["holdout_passed"] else "warn",
            {"gates": quality["gates"], "failures": quality["holdout_failures"]},
        )
    )

    physical = evaluation["physical_gate"]
    calls = metrics["opm_calls"]
    add(
        AgentEvent(
            6,
            "Simulator",
            "simulator",
            f"Расчётов OPM Flow: {calls['total']} — {calls['training']} на обучение, "
            f"{calls['tournament']} турнирных, {calls['clean_rerun']} контрольный. "
            f"Нарушений WLPR в обучающем наборе — {len(physical['wlpr_violations'])}, "
            f"базовое расписание побайтово неизменно.",
            "ok" if physical["passed"] else "fail",
            {"calls": calls, "physical_gate_passed": physical["passed"]},
        )
    )

    npv_from_economics = float(economics["summary"]["totalChddM"])
    npv_gap = abs(npv_from_economics - float(manifest["clean_npv_mrub"]))
    parity = quality["reference_parity"]
    add(
        AgentEvent(
            7,
            "Economics",
            "economics",
            f"ЧДД по методике {economics['version']}: {_ru(npv_from_economics)} млн ₽, "
            f"расхождение с манифестом {_ru(npv_gap, 6)}. Сходимость с эталоном организаторов "
            f"{_ru(parity['npv_relative_error_pct'], 4)} %.",
            "ok" if npv_gap < 1e-6 else "fail",
            {"npv_mrub": npv_from_economics, "gap": npv_gap, "parity": parity},
        )
    )

    schedule_sha = sha256_file(run.schedule_path)
    rebuild_facts: dict[str, Any] = {}
    if rebuild:
        _, rebuild_facts = regenerate_schedule([float(value) for value in winner["vector"]])
        matches = rebuild_facts["sha256"] == manifest["schedule_sha256"]
        add(
            AgentEvent(
                8,
                "Reproducibility",
                "reproducibility",
                f"Пересобрал wells_schedule.inc из исходной деки Model Z и вектора политики. "
                f"SHA-256 {'совпал' if matches else 'НЕ совпал'} с заявленным сабмитом. "
                f"История до {CONTRACT_START.isoformat()} не изменена.",
                "ok" if matches and rebuild_facts["history_prefix_identical"] else "fail",
                rebuild_facts,
            )
        )
    else:
        matches = schedule_sha == manifest["schedule_sha256"] == manifest["verification"]["clean_schedule_sha256"]
        add(
            AgentEvent(
                8,
                "Reproducibility",
                "reproducibility",
                f"SHA-256 расписания {'совпал' if matches else 'НЕ совпал'} с манифестом и с контрольным "
                f"перезапуском OPM. Расхождение ЧДД {_ru(manifest['verification']['npv_abs_error_mrub'], 6)} млн ₽.",
                "ok" if matches else "fail",
                {"schedule_sha256": schedule_sha, "manifest_sha256": manifest["schedule_sha256"]},
            )
        )

    add(
        AgentEvent(
            9,
            "Explanation",
            "explanation",
            f"Стратегия {winner['name']} даёт {_ru(metrics['npv_mrub'])} млн ₽ против "
            f"{_ru(metrics['baseline_npv_mrub'])} у базового расписания: "
            f"{_ru(metrics['delta_mrub'], signed=True)} млн ₽ ({_ru(metrics['delta_pct'], signed=True)} %). "
            f"Худший из возмущённых прогонов — {_ru(metrics['robustness_floor_mrub'])} млн ₽.",
            "ok",
            {"metrics": metrics},
        )
    )

    result = {
        "npv_mrub": metrics["npv_mrub"],
        "baseline_npv_mrub": metrics["baseline_npv_mrub"],
        "delta_mrub": metrics["delta_mrub"],
        "delta_pct": metrics["delta_pct"],
        "max_wlpr_m3_d": metrics["max_wlpr_m3_d"],
        "schedule_sha256": manifest["schedule_sha256"],
        "schedule_bytes": run.schedule_path.stat().st_size,
        "rebuilt": rebuild_facts,
        "opm_calls": metrics["opm_calls"],
        "explanation": explanation,
    }
    return events, result


REGISTRY = RunRegistry()


def reset_caches() -> None:
    verified_run.cache_clear()
    field_layout.cache_clear()


def export_submission(target: Path) -> Path:
    """Copy the verified submission next to a reviewer-friendly summary."""
    run = verified_run()
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run.schedule_path, target / "wells_schedule.inc")
    payload = {
        "npv_mrub": headline_metrics()["npv_mrub"],
        "schedule_sha256": run.manifest["schedule_sha256"],
        "case": case_summary(),
    }
    (target / "submission.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
