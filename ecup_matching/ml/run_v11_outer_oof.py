from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecup_matching.ml.data_subset import select_items_by_ids
from ecup_matching.ml.v5_evaluation import macro_ap_report
from ecup_matching.ml.v7_frozen_split import load_frozen_split_manifest, validate_frozen_split_against_matches
from ecup_matching.ml.v8_graph import graph_features, graph_rescore
from ecup_matching.ml.v11_fastlex import build_fast_pair_features
from ecup_matching.ml.v11_sparse import SparseConfig, sparse_pair_scores
from ecup_matching.ml.v11_stack import crossfit_hgb_scores


GRAPH_CONFIG = dict(
    reciprocal_best_bonus=0.0,
    reciprocal_top3_bonus=0.0,
    endpoint_rank_weight=0.02,
    ambiguity_penalty=0.01,
)


def official_pair_categories(items: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    category_by_id = items.set_index("id")["category"].astype(str)
    left = pairs["id1"].map(category_by_id)
    right = pairs["id2"].map(category_by_id)
    if left.isna().any() or right.isna().any():
        raise RuntimeError("failed to attach official pair categories")
    left_values = left.astype(str).to_numpy()
    right_values = right.astype(str).to_numpy()
    if not np.array_equal(left_values, right_values):
        raise RuntimeError("pair endpoints disagree on official category")
    return left_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    matches = pd.read_parquet(args.matches, columns=["id1", "id2", "target"])
    manifest = load_frozen_split_manifest()
    validate_frozen_split_against_matches(matches, manifest)
    dev_rows: list[int] = []
    fold_ids: list[int] = []
    for fold, rows in enumerate(manifest["fold_rows"]):
        dev_rows.extend(rows)
        fold_ids.extend([fold] * len(rows))
    dev = matches.iloc[dev_rows].reset_index(drop=True)
    folds = np.asarray(fold_ids, dtype=np.int8)
    target = dev["target"].to_numpy(np.int8)
    needed_ids = pd.unique(pd.concat([dev["id1"], dev["id2"]], ignore_index=True))
    items = select_items_by_ids(args.items, needed_ids, include_attributes=True)
    official_categories = official_pair_categories(items, dev[["id1", "id2"]])
    t_load = time.perf_counter() - started
    print(f"[v11-oof] rows={len(dev):,} items={len(items):,} load={t_load:.3f}s", flush=True)

    t = time.perf_counter()
    features = build_fast_pair_features(items, dev[["id1", "id2"]])
    features["category"] = official_categories
    fast_seconds = time.perf_counter() - t
    print(f"[v11-oof] fastlex={fast_seconds:.3f}s", flush=True)

    t = time.perf_counter()
    sparse = sparse_pair_scores(items, dev[["id1", "id2"]], config=SparseConfig(n_features=65536))
    sparse_seconds = time.perf_counter() - t
    features["sparse_cosine"] = sparse
    print(f"[v11-oof] sparse={sparse_seconds:.3f}s", flush=True)

    t = time.perf_counter()
    base = crossfit_hgb_scores(features, target, folds, min_local_rows=1200, local_blend=0.35)
    fit_seconds = time.perf_counter() - t
    work = dev[["id1", "id2", "target"]].copy()
    work["category"] = official_categories
    base_report = macro_ap_report(work, base, strict_official=True)
    base_ap = float(base_report["macro_average_precision"])
    print(f"[v11-oof] base_ap={base_ap:.12f} fit={fit_seconds:.3f}s", flush=True)

    graph = np.empty_like(base)
    fold_reports = []
    for fold in sorted(np.unique(folds).tolist()):
        mask = folds == fold
        fold_work = work.loc[mask].reset_index(drop=True)
        fold_base = base[mask]
        gf = graph_features(fold_work[["id1", "id2", "category"]], fold_base)
        fold_graph = graph_rescore(fold_base, gf, **GRAPH_CONFIG)
        graph[mask] = fold_graph
        b = float(macro_ap_report(fold_work, fold_base)["macro_average_precision"])
        g = float(macro_ap_report(fold_work, fold_graph)["macro_average_precision"])
        fold_reports.append({"fold": int(fold), "base": b, "graph": g, "delta": g - b})
    graph_report = macro_ap_report(work, graph, strict_official=True)
    graph_ap = float(graph_report["macro_average_precision"])
    total_seconds = time.perf_counter() - started

    summary = {
        "version": "v11-fastlex-1",
        "rows": len(dev),
        "items": len(items),
        "strict_oof": base_ap,
        "graph_oof": graph_ap,
        "graph_delta": graph_ap - base_ap,
        "folds": fold_reports,
        "timing_seconds": {
            "load": t_load,
            "fastlex": fast_seconds,
            "sparse": sparse_seconds,
            "crossfit": fit_seconds,
            "total": total_seconds,
        },
        "sparse_config": {"n_features": 65536, "ngram_range": [1, 2]},
        "graph_config": GRAPH_CONFIG,
        "sealed_gold_evaluated": False,
    }
    print("V11_OOF=" + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    pd.DataFrame({
        "row_index": np.asarray(dev_rows, dtype=np.int64),
        "fold": folds,
        "score": base,
        "graph_score": graph,
    }).to_parquet(args.output_dir / "v11-oof.parquet", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
