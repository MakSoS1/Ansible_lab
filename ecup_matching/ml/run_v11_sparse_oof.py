from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ecup_matching.ml.data_subset import select_items_by_ids
from ecup_matching.ml.v5_evaluation import macro_ap_report
from ecup_matching.ml.v7_frozen_split import load_frozen_split_manifest, validate_frozen_split_against_matches
from ecup_matching.ml.v8_graph import graph_features, graph_rescore
from ecup_matching.ml.v11_sparse_channels import build_sparse_pair_features
from ecup_matching.ml.v11_stack import crossfit_hgb_scores

PREVALENCE_RATIO = 0.566880890615799
GRAPH_CONFIG = dict(reciprocal_best_bonus=0.0, reciprocal_top3_bonus=0.0, endpoint_rank_weight=0.02, ambiguity_penalty=0.01)


def stress(work: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    y = work.target.to_numpy(np.int8)
    cats = work.category.astype(str).to_numpy()
    categories = sorted(np.unique(cats).tolist())
    plan = {}
    for cat in categories:
        neg = np.flatnonzero((cats == cat) & (y == 0))
        pos = np.flatnonzero((cats == cat) & (y == 1))
        prevalence = float(len(pos) / max(1, len(pos) + len(neg)))
        shifted = prevalence * PREVALENCE_RATIO
        keep = min(len(pos), max(1, int(round(shifted * len(neg) / max(1e-12, 1.0 - shifted)))))
        plan[cat] = (neg, pos, keep)
    values = []
    for seed in range(2026, 2126):
        rng = np.random.default_rng(seed)
        aps = []
        for cat in categories:
            neg, pos, keep = plan[cat]
            selected = np.concatenate([neg, rng.choice(pos, size=keep, replace=False)])
            aps.append(float(average_precision_score(y[selected], scores[selected])))
        values.append(float(np.mean(aps)))
    arr = np.asarray(values)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)), "p05": float(np.quantile(arr, .05)), "p50": float(np.quantile(arr, .5)), "p95": float(np.quantile(arr, .95))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matches", type=Path, required=True)
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    matches = pd.read_parquet(a.matches, columns=["id1", "id2", "target"])
    manifest = load_frozen_split_manifest()
    validate_frozen_split_against_matches(matches, manifest)
    rows, fold_ids = [], []
    for fold, fold_rows in enumerate(manifest["fold_rows"]):
        rows.extend(fold_rows)
        fold_ids.extend([fold] * len(fold_rows))
    dev = matches.iloc[rows].reset_index(drop=True)
    folds = np.asarray(fold_ids, dtype=np.int8)
    wanted = pd.unique(pd.concat([dev.id1, dev.id2], ignore_index=True))
    items = select_items_by_ids(a.items, wanted, include_attributes=True)
    load_seconds = time.perf_counter() - started
    t = time.perf_counter()
    features = build_sparse_pair_features(items, dev[["id1", "id2"]])
    feature_seconds = time.perf_counter() - t
    print(f"[v11-sparse] rows={len(dev):,} items={len(items):,} features={feature_seconds:.3f}s", flush=True)
    t = time.perf_counter()
    base = crossfit_hgb_scores(features, dev.target.to_numpy(np.int8), folds, min_local_rows=1200, local_blend=0.35)
    fit_seconds = time.perf_counter() - t
    work = dev.copy()
    work["category"] = features.category.astype(str).to_numpy()
    base_ap = float(macro_ap_report(work, base, strict_official=True)["macro_average_precision"])
    graph = np.empty_like(base)
    fold_reports = []
    for fold in sorted(np.unique(folds).tolist()):
        mask = folds == fold
        fw = work.loc[mask].reset_index(drop=True)
        bs = base[mask]
        gf = graph_features(fw[["id1", "id2", "category"]], bs)
        gs = graph_rescore(bs, gf, **GRAPH_CONFIG)
        graph[mask] = gs
        b = float(macro_ap_report(fw, bs)["macro_average_precision"])
        g = float(macro_ap_report(fw, gs)["macro_average_precision"])
        fold_reports.append({"fold": int(fold), "base": b, "graph": g, "delta": g-b})
    graph_ap = float(macro_ap_report(work, graph, strict_official=True)["macro_average_precision"])
    summary = {
        "version": "v11-sparse-channels-1",
        "rows": len(dev),
        "items": len(items),
        "strict_oof": base_ap,
        "graph_oof": graph_ap,
        "graph_delta": graph_ap-base_ap,
        "stress_base": stress(work, base),
        "stress_graph": stress(work, graph),
        "prevalence_ratio": PREVALENCE_RATIO,
        "folds": fold_reports,
        "timing_seconds": {"load": load_seconds, "features": feature_seconds, "crossfit": fit_seconds, "total": time.perf_counter()-started},
        "sealed_gold_evaluated": False,
    }
    print("V11_SPARSE_OOF=" + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    pd.DataFrame({"row_index": np.asarray(rows, np.int64), "fold": folds, "score": base, "graph_score": graph}).to_parquet(a.output_dir / "v11-sparse-oof.parquet", index=False)
    (a.output_dir / "sparse-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
