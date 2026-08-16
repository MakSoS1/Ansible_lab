"""Two-hop support for candidate edges — the piece `v8_graph` never had.

`v8_graph` computes endpoint degree, endpoint rank and reciprocal-best flags.
It does not look at triangles at all, so it cannot express the property that
makes product identity different from independent pair classification:

    A = B  and  B = C   =>   A = C

For an edge ``(a, b)`` the support is the strength of the best two-hop path
through a shared neighbour ``c``: ``max_c min(s(a,c), s(b,c))``. A pair with a
strong two-hop path is corroborated by the rest of the graph; a pair with a
high direct score and no support is a candidate false edge.

Everything here is target-free and scoped per category, so it may be combined
with the existing percentile-rank machinery without touching labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def transitive_features(frame: pd.DataFrame, scores) -> pd.DataFrame:
    """Two-hop corroboration for every observed edge.

    Returns ``two_hop_support`` (best bottleneck score over shared neighbours),
    ``shared_neighbours`` (their count) and ``support_minus_direct`` (support
    relative to the edge's own score). Edges with no shared neighbour get
    support ``0.0``, which is distinguishable from a weak path because
    ``shared_neighbours`` is then ``0``.
    """
    required = {"id1", "id2", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"transitive frame missing columns: {sorted(missing)}")
    score = np.asarray(scores, dtype=np.float64)
    if score.ndim != 1 or len(score) != len(frame):
        raise ValueError("scores must be one-dimensional and aligned with frame")
    if len(score) == 0:
        raise ValueError("transitive scorer requires at least one edge")
    if not np.isfinite(score).all():
        raise ValueError("scores contain NaN or infinity")

    categories = frame["category"].astype(str).to_numpy()
    left = frame["id1"].to_numpy()
    right = frame["id2"].to_numpy()

    support = np.zeros(len(frame), dtype=np.float64)
    shared = np.zeros(len(frame), dtype=np.int32)

    for category in pd.unique(categories):
        rows = np.flatnonzero(categories == category)
        # Adjacency inside one category only: an identifier reused across
        # categories must not create a path between them.
        neighbours: dict[object, dict[object, float]] = {}
        for row in rows:
            a, b, s = left[row], right[row], score[row]
            # Keep the strongest observed edge when a pair repeats.
            if neighbours.setdefault(a, {}).get(b, -np.inf) < s:
                neighbours[a][b] = s
            if neighbours.setdefault(b, {}).get(a, -np.inf) < s:
                neighbours[b][a] = s
        for row in rows:
            a, b = left[row], right[row]
            na, nb = neighbours[a], neighbours[b]
            if len(nb) < len(na):
                na, nb = nb, na
            best = 0.0
            count = 0
            for node, s_first in na.items():
                if node == a or node == b:
                    continue
                s_second = nb.get(node)
                if s_second is None:
                    continue
                count += 1
                bottleneck = s_first if s_first < s_second else s_second
                if bottleneck > best:
                    best = bottleneck
            support[row] = best
            shared[row] = count

    out = pd.DataFrame(
        {
            "two_hop_support": support,
            "shared_neighbours": shared,
            "has_two_hop": (shared > 0).astype(float),
            "support_minus_direct": support - score,
        }
    )
    if not np.isfinite(out.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("transitive feature computation produced nonfinite values")
    return out


def graph_degree_report(frame: pd.DataFrame) -> dict[str, object]:
    """Describe the candidate graph before trusting any graph feature.

    If almost every item has exactly one incident edge then `reciprocal_best`
    is trivially true everywhere and carries no information — which would make
    a zero weight the correct answer on this graph, and say nothing about a
    real retrieval graph.
    """
    required = {"id1", "id2", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"degree frame missing columns: {sorted(missing)}")
    categories = frame["category"].astype(str).to_numpy()
    endpoint = pd.DataFrame(
        {
            "category": np.concatenate([categories, categories]),
            "item": np.concatenate([frame["id1"].to_numpy(), frame["id2"].to_numpy()]),
        }
    )
    degree = endpoint.groupby(["category", "item"], sort=False).size()
    values = degree.to_numpy(dtype=np.float64)
    per_category = (
        degree.reset_index(name="degree")
        .groupby("category")["degree"]
        .agg(["size", "mean", "max"])
        .rename(columns={"size": "items"})
    )
    return {
        "edges": int(len(frame)),
        "items": int(len(values)),
        "degree_mean": float(values.mean()),
        "degree_median": float(np.median(values)),
        "degree_p90": float(np.quantile(values, 0.90)),
        "degree_max": float(values.max()),
        "fraction_degree_1": float((values <= 1).mean()),
        "fraction_degree_ge_3": float((values >= 3).mean()),
        "per_category": {
            str(name): {
                "items": int(row["items"]),
                "degree_mean": float(row["mean"]),
                "degree_max": float(row["max"]),
            }
            for name, row in per_category.iterrows()
        },
    }


def rescore_with_transitivity(
    base_percentile,
    graph_frame: pd.DataFrame,
    transitive: pd.DataFrame,
    *,
    support_weight: float = 0.0,
    orphan_penalty: float = 0.0,
) -> np.ndarray:
    """Add two-hop corroboration on top of an existing graph score.

    ``support_weight`` rewards an edge whose two-hop path is strong relative to
    its own score; ``orphan_penalty`` shrinks an edge that has no shared
    neighbour at all. Both default to zero so the caller must opt in.
    """
    out = np.asarray(base_percentile, dtype=np.float64).copy()
    if out.ndim != 1 or len(out) != len(transitive):
        raise ValueError("base score and transitive features must be aligned")
    if len(graph_frame) != len(transitive):
        raise ValueError("graph frame and transitive features must be aligned")
    out += float(support_weight) * transitive["two_hop_support"].to_numpy(float)
    out -= float(orphan_penalty) * (1.0 - transitive["has_two_hop"].to_numpy(float))
    if not np.isfinite(out).all():
        raise RuntimeError("transitive rescore produced nonfinite values")
    return out


__all__ = [
    "graph_degree_report",
    "rescore_with_transitivity",
    "transitive_features",
]
