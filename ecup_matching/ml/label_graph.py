from __future__ import annotations

from collections import defaultdict
from itertools import combinations

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


def _required(df: pd.DataFrame) -> None:
    missing = {"id1", "id2", "target"} - set(df.columns)
    if missing:
        raise ValueError(f"pairs missing required columns: {sorted(missing)}")


def _ordered_pair(a: object, b: object) -> tuple[object, object]:
    try:
        return (a, b) if a <= b else (b, a)
    except TypeError:
        ka, kb = (type(a).__name__, repr(a)), (type(b).__name__, repr(b))
        return (a, b) if ka <= kb else (b, a)


def canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize pair direction without changing row order or extra columns."""
    _required(df)
    out = df.copy()
    ordered = [_ordered_pair(a, b) for a, b in out[["id1", "id2"]].itertuples(index=False, name=None)]
    out["id1"] = [p[0] for p in ordered]
    out["id2"] = [p[1] for p in ordered]
    return out


def clean_human_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove duplicate rows and drop entire human pairs with contradictory labels."""
    canon = canonicalize_pairs(df).reset_index(drop=True)
    grouped = canon.groupby(["id1", "id2"], sort=False, dropna=False)
    conflicting_keys: set[tuple[object, object]] = set()
    duplicate_rows_removed = 0
    keep_rows: list[int] = []

    for key, group in grouped:
        labels = pd.unique(group["target"])
        if len(labels) > 1:
            conflicting_keys.add(key)
            continue
        keep_rows.append(int(group.index[0]))
        duplicate_rows_removed += max(0, len(group) - 1)

    out = canon.loc[keep_rows].reset_index(drop=True)
    report = {
        "input_rows": int(len(canon)),
        "output_rows": int(len(out)),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "conflicting_pairs_dropped": int(len(conflicting_keys)),
        "conflicting_rows_dropped": int(
            sum(len(grouped.get_group(key)) for key in conflicting_keys)
        ),
    }
    return out, report


def positive_components(df: pd.DataFrame) -> dict[object, object]:
    """Map items participating in positive human edges to a deterministic component id."""
    _required(df)
    uf = _UnionFind()
    positives = canonicalize_pairs(df)
    positives = positives[positives["target"].astype(float) >= 0.5]
    for a, b in positives[["id1", "id2"]].itertuples(index=False, name=None):
        uf.union(a, b)

    groups: dict[object, list[object]] = defaultdict(list)
    for item in uf.parent:
        groups[uf.find(item)].append(item)

    result: dict[object, object] = {}
    for members in groups.values():
        # Component identifier only needs equality semantics; repr ordering also
        # works for heterogeneous IDs while staying deterministic.
        component_id = min(members, key=lambda x: (type(x).__name__, repr(x)))
        for item in members:
            result[item] = component_id
    return result


def augment_transitive_positives(
    df: pd.DataFrame,
    max_pairs_per_component: int = 2000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add capped positive closure edges while treating explicit negatives as vetoes."""
    if max_pairs_per_component < 0:
        raise ValueError("max_pairs_per_component must be non-negative")
    clean, clean_report = clean_human_pairs(df)
    components = positive_components(clean)

    members_by_component: dict[object, list[object]] = defaultdict(list)
    for item, component in components.items():
        members_by_component[component].append(item)

    existing_positive: set[tuple[object, object]] = set()
    explicit_negative: set[tuple[object, object]] = set()
    for a, b, target in clean[["id1", "id2", "target"]].itertuples(index=False, name=None):
        pair = _ordered_pair(a, b)
        if float(target) >= 0.5:
            existing_positive.add(pair)
        else:
            explicit_negative.add(pair)

    generated: list[dict[str, object]] = []
    for component in sorted(members_by_component, key=lambda x: (type(x).__name__, repr(x))):
        members = sorted(members_by_component[component], key=lambda x: (type(x).__name__, repr(x)))
        added_here = 0
        for a, b in combinations(members, 2):
            pair = _ordered_pair(a, b)
            if pair in existing_positive or pair in explicit_negative:
                continue
            if added_here >= max_pairs_per_component:
                break
            generated.append({"id1": pair[0], "id2": pair[1], "target": 1})
            existing_positive.add(pair)
            added_here += 1

    if generated:
        extra = pd.DataFrame(generated)
        # Preserve optional columns in the original frame with missing values.
        for column in clean.columns:
            if column not in extra.columns:
                extra[column] = pd.NA
        extra = extra[clean.columns]
        out = pd.concat([clean, extra], ignore_index=True)
    else:
        out = clean.copy()

    report = dict(clean_report)
    report.update(
        {
            "positive_components": int(len(members_by_component)),
            "transitive_positive_rows_added": int(len(generated)),
            "explicit_negative_veto_pairs": int(len(explicit_negative)),
            "augmented_rows": int(len(out)),
        }
    )
    return out, report
