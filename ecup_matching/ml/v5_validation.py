from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[object, object] = {}
        self.rank: dict[object, int] = {}

    def add(self, value: object) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: object) -> object:
        self.add(value)
        root = self.parent[value]
        if root != value:
            self.parent[value] = self.find(root)
        return self.parent[value]

    def union(self, left: object, right: object) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _component_rows(matches: pd.DataFrame) -> list[list[int]]:
    required = {"id1", "id2"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches missing required columns: {sorted(missing)}")
    if len(matches) < 2:
        raise ValueError("at least two pair rows are required")

    uf = _UnionFind()
    for id1, id2 in matches[["id1", "id2"]].itertuples(index=False, name=None):
        uf.union(id1, id2)

    rows_by_root: dict[object, list[int]] = defaultdict(list)
    for row_idx, id1 in enumerate(matches["id1"].tolist()):
        rows_by_root[uf.find(id1)].append(row_idx)
    return [sorted(rows) for rows in rows_by_root.values()]


def _descriptor_matrix(descriptors: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if len(descriptors) == 0:
        raise ValueError("descriptors must not be empty")
    columns: list[np.ndarray] = [np.ones(len(descriptors), dtype=np.float64)]
    names = ["__rows__"]
    for column in descriptors.columns:
        values = descriptors[column].astype("string").fillna("__missing__")
        for value in sorted(values.unique().tolist()):
            columns.append((values == value).to_numpy(dtype=np.float64))
            names.append(f"{column}={value}")
    return np.column_stack(columns), names


def _balanced_component_assignment(
    components: list[list[int]],
    descriptor_matrix: np.ndarray,
    proportions: np.ndarray,
    seed: int,
) -> list[list[int]]:
    totals = descriptor_matrix.sum(axis=0)
    targets = proportions[:, None] * totals[None, :]
    current = np.zeros_like(targets, dtype=np.float64)
    assigned: list[list[int]] = [[] for _ in proportions]

    rng = np.random.default_rng(seed)
    tie_break = rng.random(len(components))
    order = sorted(
        range(len(components)),
        key=lambda i: (-len(components[i]), float(tie_break[i])),
    )

    # Row count is the first feature. Give it extra weight so folds remain
    # practical in size while categorical descriptors refine the assignment.
    feature_weights = np.ones(descriptor_matrix.shape[1], dtype=np.float64)
    feature_weights[0] = 3.0

    for component_idx in order:
        rows = components[component_idx]
        vector = descriptor_matrix[rows].sum(axis=0)
        candidate_scores: list[float] = []
        for split_id in range(len(proportions)):
            candidate = current.copy()
            candidate[split_id] += vector
            scale = np.maximum(targets, 1.0)
            normalized_error = ((candidate - targets) / scale) ** 2
            score = float((normalized_error * feature_weights[None, :]).sum())

            # Strongly discourage assigning beyond a split's target while
            # another split is still materially under target.
            target_rows = max(targets[split_id, 0], 1.0)
            row_ratio = candidate[split_id, 0] / target_rows
            if row_ratio > 1.15:
                score += float((row_ratio - 1.15) ** 2 * 100.0)
            candidate_scores.append(score)

        chosen = min(range(len(proportions)), key=lambda i: (candidate_scores[i], i))
        current[chosen] += vector
        assigned[chosen].extend(rows)

    # A tiny synthetic dataset can otherwise leave a split empty. Move one
    # whole smallest component from the largest populated split when possible.
    for empty_id, rows in enumerate(assigned):
        if rows:
            continue
        donors = [i for i, donor_rows in enumerate(assigned) if len(donor_rows) > 1]
        if not donors:
            raise ValueError("not enough connected components for requested splits")
        donor = max(donors, key=lambda i: len(assigned[i]))
        donor_set = set(assigned[donor])
        donor_components = [c for c in components if set(c).issubset(donor_set)]
        movable = min(donor_components, key=lambda c: (len(c), c[0]))
        move = set(movable)
        assigned[donor] = [row for row in assigned[donor] if row not in move]
        assigned[empty_id].extend(movable)

    return [sorted(rows) for rows in assigned]


def build_v5_split_manifest(
    matches: pd.DataFrame,
    descriptors: pd.DataFrame,
    *,
    gold_fraction: float = 0.22,
    n_folds: int = 5,
    seed: int = 2026,
) -> dict[str, Any]:
    """Build deterministic item-component-disjoint gold + development folds."""
    if len(matches) != len(descriptors):
        raise ValueError("matches and descriptors must contain the same number of rows")
    if not 0.0 < gold_fraction < 1.0:
        raise ValueError("gold_fraction must be between 0 and 1")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    components = _component_rows(matches)
    if len(components) < n_folds + 1:
        raise ValueError("not enough connected components for gold plus requested folds")

    matrix, descriptor_names = _descriptor_matrix(descriptors.reset_index(drop=True))
    proportions = np.asarray(
        [gold_fraction] + [(1.0 - gold_fraction) / n_folds] * n_folds,
        dtype=np.float64,
    )
    assigned = _balanced_component_assignment(components, matrix, proportions, seed)

    manifest: dict[str, Any] = {
        "version": 1,
        "seed": int(seed),
        "gold_fraction": float(gold_fraction),
        "n_folds": int(n_folds),
        "row_count": int(len(matches)),
        "component_count": int(len(components)),
        "descriptor_names": descriptor_names,
        "gold_rows": assigned[0],
        "fold_rows": assigned[1:],
    }
    report = validate_manifest_no_overlap(matches, manifest)
    if report["duplicate_rows"] or report["cross_split_item_overlap"]:
        raise RuntimeError(f"invalid v5 split manifest: {report}")
    return manifest


def validate_manifest_no_overlap(matches: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, int]:
    splits = [list(manifest.get("gold_rows", [])), *[list(x) for x in manifest.get("fold_rows", [])]]
    flat = [int(row) for rows in splits for row in rows]
    if any(row < 0 or row >= len(matches) for row in flat):
        raise IndexError("manifest contains an out-of-range row index")

    duplicate_rows = len(flat) - len(set(flat))
    item_sets: list[set[object]] = []
    for rows in splits:
        subset = matches.iloc[rows]
        item_sets.append(set(subset["id1"].tolist()) | set(subset["id2"].tolist()))

    overlap_items: set[object] = set()
    for left in range(len(item_sets)):
        for right in range(left + 1, len(item_sets)):
            overlap_items.update(item_sets[left] & item_sets[right])

    return {
        "row_coverage": int(len(set(flat))),
        "duplicate_rows": int(duplicate_rows),
        "missing_rows": int(len(matches) - len(set(flat))),
        "cross_split_item_overlap": int(len(overlap_items)),
    }


def manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
