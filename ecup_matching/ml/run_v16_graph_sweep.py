"""Sweep the graph rescoring parameters that were never searched.

`ecup-v8-gate55-graph-tune.yml` searched `endpoint_rank_weight` and
`ambiguity_penalty` while holding `reciprocal_best_bonus` and
`reciprocal_top3_bonus` pinned at `0.0`. The reciprocal signal was therefore
never evaluated, and the recorded `graph_delta` of `+0.0015` describes only the
two parameters that were varied.

This driver searches all four, adds the two-hop terms that did not exist, and
reports the degree distribution first — because on a graph where nearly every
item has one incident edge, `reciprocal_best` is constant and a zero weight is
the right answer for that graph while saying nothing about a retrieval graph.

Features are computed inside each held fold, never across folds, so no fold
sees another fold's edges. Labels are used only to score the result.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .progress import ProgressReporter
from .v5_evaluation import macro_ap_report
from .v8_graph import graph_features, graph_rescore
from .v16_graph_transitive import (
    graph_degree_report,
    rescore_with_transitivity,
    transitive_features,
)


RECIPROCAL_BEST_GRID = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)
RECIPROCAL_TOP3_GRID = (0.0, 0.0025, 0.005, 0.01, 0.02)
ENDPOINT_RANK_GRID = (0.0, 0.01, 0.02, 0.04)
AMBIGUITY_GRID = (0.0, 0.005, 0.01, 0.02)
SUPPORT_GRID = (0.0, 0.01, 0.02, 0.04, 0.08)
ORPHAN_GRID = (0.0, 0.005, 0.01, 0.02)


def _load_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "row_index" not in frame.columns:
        raise ValueError(f"{path} has no row_index column; found {list(frame.columns)}")
    numeric = [
        column
        for column in frame.columns
        if column not in {"row_index", "fold", "target", "category", "id1", "id2"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    named = [c for c in numeric if "score" in c.lower() or "predict" in c.lower()]
    column = named[0] if len(named) == 1 else (numeric[0] if numeric else None)
    if column is None:
        raise ValueError(f"{path} has no usable score column")
    out = frame[["row_index", column]].rename(columns={column: "score"}).copy()
    out["row_index"] = out["row_index"].astype(np.int64)
    if "fold" in frame.columns:
        out["fold"] = frame["fold"].astype(np.int16)
    else:
        out["fold"] = np.int16(0)
    return out


def run_graph_sweep(
    *,
    scores_path: Path,
    human_items_path: Path,
    human_matches_path: Path,
    output_path: Path,
    baseline_macro_ap: float | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    scores = _load_scores(Path(scores_path))
    matches = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    items = pd.read_parquet(human_items_path, columns=["id", "category"]).drop_duplicates("id")
    category_by_id = dict(zip(items["id"].tolist(), items["category"].astype(str).tolist()))

    work = matches.iloc[scores["row_index"].to_numpy()].reset_index(drop=True)
    work["category"] = work["id1"].map(category_by_id)
    if work["category"].isna().any():
        raise RuntimeError("category mapping is incomplete for the scored rows")
    right = work["id2"].map(category_by_id).astype(str)
    if not np.array_equal(work["category"].astype(str).to_numpy(), right.to_numpy()):
        raise RuntimeError("scored set contains a cross-category pair")
    work["fold"] = scores["fold"].to_numpy()
    base = scores["score"].to_numpy(dtype=np.float64)

    degree = graph_degree_report(work)
    base_ap = float(macro_ap_report(work, base)["macro_average_precision"])
    if baseline_macro_ap is not None and abs(base_ap - float(baseline_macro_ap)) > 1e-9:
        raise RuntimeError(
            f"baseline reproduction failed: expected {baseline_macro_ap}, got {base_ap}"
        )
    print(
        json.dumps(
            {"phase": "graph-degree", "base_macro_ap": base_ap, **degree},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    # Fold-local features: a held fold never sees another fold's edges.
    folds = sorted(pd.unique(work["fold"]).tolist())
    feature_by_fold: dict[int, pd.DataFrame] = {}
    transitive_by_fold: dict[int, pd.DataFrame] = {}
    for fold in folds:
        mask = (work["fold"] == fold).to_numpy()
        part = work.loc[mask].reset_index(drop=True)
        feature_by_fold[int(fold)] = graph_features(part, base[mask])
        transitive_by_fold[int(fold)] = transitive_features(part, base[mask])

    combos = list(
        itertools.product(
            RECIPROCAL_BEST_GRID,
            RECIPROCAL_TOP3_GRID,
            ENDPOINT_RANK_GRID,
            AMBIGUITY_GRID,
            SUPPORT_GRID,
            ORPHAN_GRID,
        )
    )
    progress = ProgressReporter(
        "graph-sweep", len(combos), every_units=200, every_seconds=30.0
    )
    results: list[dict[str, float]] = []
    for index, (rb, rt, ep, ap, sw, orp) in enumerate(combos, start=1):
        score = np.empty(len(work), dtype=np.float64)
        for fold in folds:
            mask = (work["fold"] == fold).to_numpy()
            rescored = graph_rescore(
                base[mask],
                feature_by_fold[int(fold)],
                reciprocal_best_bonus=rb,
                reciprocal_top3_bonus=rt,
                endpoint_rank_weight=ep,
                ambiguity_penalty=ap,
            )
            score[mask] = rescore_with_transitivity(
                rescored,
                work.loc[mask].reset_index(drop=True),
                transitive_by_fold[int(fold)],
                support_weight=sw,
                orphan_penalty=orp,
            )
        macro = float(macro_ap_report(work, score)["macro_average_precision"])
        results.append(
            {
                "rb": rb, "rt": rt, "ep": ep, "ap": ap,
                "support": sw, "orphan": orp,
                "macro_ap": macro, "delta": macro - base_ap,
            }
        )
        progress.update(index)
    progress.finish(len(combos))

    ranked = sorted(results, key=lambda r: (-r["macro_ap"], r["rb"] + r["rt"] + r["ep"] + r["ap"]))
    best = ranked[0]
    zero_config = next(
        r for r in results
        if r["rb"] == 0.0 and r["rt"] == 0.0 and r["support"] == 0.0 and r["orphan"] == 0.0
        and r["ep"] == 0.02 and r["ap"] == 0.01
    )
    payload = {
        "version": "v16-graph-sweep",
        "diagnostic_only": True,
        "scores_source": str(scores_path),
        "rows": int(len(work)),
        "folds": [int(f) for f in folds],
        "base_macro_ap": base_ap,
        "degree_report": degree,
        "best": best,
        "v9_frozen_config_for_reference": zero_config,
        "best_minus_v9_frozen": best["macro_ap"] - zero_config["macro_ap"],
        "reciprocal_ever_helps": any(r["delta"] > 0 and r["rb"] > 0 for r in results),
        "transitivity_ever_helps": any(r["delta"] > 0 and r["support"] > 0 for r in results),
        "top_10": ranked[:10],
        "combinations": len(combos),
        "elapsed_seconds": time.perf_counter() - started,
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--human-items", type=Path, required=True)
    parser.add_argument("--human-matches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-macro-ap", type=float, default=None)
    args = parser.parse_args()
    run_graph_sweep(
        scores_path=args.scores,
        human_items_path=args.human_items,
        human_matches_path=args.human_matches,
        output_path=args.output,
        baseline_macro_ap=args.baseline_macro_ap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
