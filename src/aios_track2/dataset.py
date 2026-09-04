from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SplitAssignment:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioManifest:
    scenario_id: str
    seed: int
    deck_sha256: str
    schedule_sha256: str
    simulator_version: str
    status: str
    runtime_seconds: float
    npv_mrub: float
    constraint_violations: tuple[str, ...]
    github_run_url: str
    monthly_path: str
    created_at: str


def split_scenarios(ids: Sequence[str], seed: int, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> SplitAssignment:
    ordered = list(ids)
    rng_order = sorted(ordered, key=lambda item: sha256(f"{seed}:{item}".encode()).hexdigest())
    n_total = len(rng_order)
    n_train = int(n_total * fractions[0])
    n_val = int(n_total * fractions[1])
    train = tuple(rng_order[:n_train])
    validation = tuple(rng_order[n_train : n_train + n_val])
    test = tuple(rng_order[n_train + n_val :])
    if n_total >= 3:
        groups = [list(train), list(validation), list(test)]
        for index, group in enumerate(groups):
            if not group:
                donor = max(range(3), key=lambda idx: len(groups[idx]))
                groups[index].append(groups[donor].pop())
        train, validation, test = tuple(groups[0]), tuple(groups[1]), tuple(groups[2])
    return SplitAssignment(train=train, validation=validation, test=test)


def write_scenario(
    scenario_id: str,
    monthly: pd.DataFrame,
    root: Path,
    *,
    seed: int,
    deck_sha256: str,
    schedule_sha256: str,
    simulator_version: str,
    status: str,
    runtime_seconds: float,
    npv_mrub: float,
    violations: tuple[str, ...] = (),
    github_run_url: str = "",
) -> ScenarioManifest:
    folder = root / "scenarios" / scenario_id
    folder.mkdir(parents=True, exist_ok=True)
    monthly_path = folder / "monthly.parquet"
    monthly.to_parquet(monthly_path, index=False)
    manifest = ScenarioManifest(
        scenario_id=scenario_id,
        seed=seed,
        deck_sha256=deck_sha256,
        schedule_sha256=schedule_sha256,
        simulator_version=simulator_version,
        status=status,
        runtime_seconds=runtime_seconds,
        npv_mrub=npv_mrub,
        constraint_violations=violations,
        github_run_url=github_run_url,
        monthly_path=str(monthly_path.relative_to(root)),
        created_at=datetime.now(UTC).isoformat(),
    )
    payload = asdict(manifest)
    payload["monthly_sha256"] = sha256(monthly_path.read_bytes()).hexdigest()
    manifest_path = folder / "manifest.json"
    tmp = folder / "manifest.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(manifest_path)
    return manifest
