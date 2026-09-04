from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Sequence

import numpy as np
from scipy.stats import qmc

from .deck import Well

CHALLENGE_NODE_DATES = (date(2007, 1, 1), date(2016, 1, 1), date(2025, 1, 1))
CHALLENGE_PRODUCER_GROUPS = 4
CHALLENGE_INJECTOR_GROUPS = 2
CHALLENGE_GROUPS = CHALLENGE_PRODUCER_GROUPS + CHALLENGE_INJECTOR_GROUPS
CHALLENGE_DIMENSIONS = CHALLENGE_GROUPS * len(CHALLENGE_NODE_DATES)
_DESIGN_SEED = 57721
_SPLIT_SEED = 81173
_LOWER = 0.8
_UPPER = 1.2
_MAX_NODE_DELTA = 0.12
FROZEN_CHALLENGE_SHA256 = "9d64588192a01290323b431ff9f88f565666b74888bc4e9a24787cc0388846c5"


@dataclass(frozen=True, slots=True)
class ChallengeScenario:
    scenario_id: int
    split: str
    values: tuple[float, ...]

    def flat_vector(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["values"] = list(self.values)
        return payload

    def producer_nodes(self) -> dict[int, tuple[float, ...]]:
        matrix = self.flat_vector().reshape(CHALLENGE_GROUPS, len(CHALLENGE_NODE_DATES))
        return {group: tuple(float(v) for v in matrix[group]) for group in range(CHALLENGE_PRODUCER_GROUPS)}

    def injector_nodes(self) -> dict[int, tuple[float, ...]]:
        matrix = self.flat_vector().reshape(CHALLENGE_GROUPS, len(CHALLENGE_NODE_DATES))
        return {
            group: tuple(float(v) for v in matrix[CHALLENGE_PRODUCER_GROUPS + group])
            for group in range(CHALLENGE_INJECTOR_GROUPS)
        }


def frozen_challenge_doe() -> tuple[ChallengeScenario, ...]:
    sampler = qmc.Sobol(d=CHALLENGE_DIMENSIONS, scramble=True, seed=_DESIGN_SEED)
    raw = _LOWER + (_UPPER - _LOWER) * sampler.random_base2(m=6)
    matrix = raw.reshape(64, CHALLENGE_GROUPS, len(CHALLENGE_NODE_DATES))
    for node in range(1, len(CHALLENGE_NODE_DATES)):
        previous = matrix[:, :, node - 1]
        matrix[:, :, node] = np.clip(matrix[:, :, node], previous - _MAX_NODE_DELTA, previous + _MAX_NODE_DELTA)
    matrix = np.clip(matrix, _LOWER, _UPPER)
    matrix[0, :, :] = 1.0

    rng = np.random.default_rng(_SPLIT_SEED)
    permutation = rng.permutation(np.arange(1, 64)).tolist()
    train = {0, *permutation[:39]}
    validation = set(permutation[39:47])

    rows: list[ChallengeScenario] = []
    for scenario_id in range(64):
        split = "train" if scenario_id in train else "validation" if scenario_id in validation else "holdout"
        values = tuple(round(float(value), 12) for value in matrix[scenario_id].reshape(-1))
        rows.append(ChallengeScenario(scenario_id, split, values))
    return tuple(rows)


def challenge_design_sha256() -> str:
    payload = [row.as_dict() for row in frozen_challenge_doe()]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def challenge_scenario_by_id(scenario_id: int) -> ChallengeScenario:
    rows = frozen_challenge_doe()
    if not 0 <= scenario_id < len(rows):
        raise ValueError(f"scenario_id must be in [0, {len(rows) - 1}]")
    return rows[scenario_id]


def _morton_key(i: int, j: int) -> int:
    x = max(0, int(i))
    y = max(0, int(j))
    key = 0
    for bit in range(16):
        key |= ((x >> bit) & 1) << (2 * bit)
        key |= ((y >> bit) & 1) << (2 * bit + 1)
    return key


def deterministic_spatial_groups(wells: Sequence[Well], group_count: int) -> dict[str, int]:
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    unique = {well.name: well for well in wells}
    if len(unique) < group_count:
        raise ValueError("group_count cannot exceed the number of wells")
    ordered = sorted(unique.values(), key=lambda well: (_morton_key(well.i, well.j), well.i, well.j, well.name))
    n = len(ordered)
    mapping: dict[str, int] = {}
    start = 0
    for group in range(group_count):
        size = n // group_count + int(group < (n % group_count))
        for well in ordered[start : start + size]:
            mapping[well.name] = group
        start += size
    return mapping


def schedule_role_names(text: str) -> tuple[set[str], set[str]]:
    clean = re.sub(r"--[^\n]*", "", text)

    def names(keyword: str) -> set[str]:
        result: set[str] = set()
        active = False
        for raw in clean.splitlines():
            line = raw.strip()
            upper = line.upper()
            if not active:
                if upper == keyword:
                    active = True
                continue
            if line == "/":
                active = False
                continue
            match = re.match(r"['\"]([^'\"]+)['\"]", line)
            if match:
                result.add(match.group(1))
        return result

    return names("WCONPROD"), names("WCONINJE")


def legacy_policy_to_challenge_vector(
    *, producer_2007: float, producer_2025: float, injector_2007: float, injector_2025: float
) -> np.ndarray:
    producer_mid = (float(producer_2007) + float(producer_2025)) / 2.0
    injector_mid = (float(injector_2007) + float(injector_2025)) / 2.0
    producer = np.asarray([producer_2007, producer_mid, producer_2025], dtype=float)
    injector = np.asarray([injector_2007, injector_mid, injector_2025], dtype=float)
    matrix = np.vstack(
        [np.tile(producer, (CHALLENGE_PRODUCER_GROUPS, 1)), np.tile(injector, (CHALLENGE_INJECTOR_GROUPS, 1))]
    )
    return matrix.reshape(-1)
