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
from ecup_matching.ml.v11_weighted_overlap import crossfit_weighted_overlap

PREVALENCE_RATIO = 0.566880890615799
GRAPH_CONFIG = dict(reciprocal_best_bonus=0.0, reciprocal_top3_bonus=0.0, endpoint_rank_weight=0.02, ambiguity_penalty=0.01)


def stress(work: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    y = work.target.to_numpy(np.int8); cats = work.category.astype(str).to_numpy(); categories = sorted(np.unique(cats).tolist()); plan = {}
    for cat in categories:
        neg = np.flatnonzero((cats == cat) & (y == 0)); pos = np.flatnonzero((cats == cat) & (y == 1)); prevalence = float(len(pos) / max(1, len(pos) + len(neg))); shifted = prevalence * PREVALENCE_RATIO
        keep = min(len(pos), max(1, int(round(shifted * len(neg) / max(1e-12, 1.0 - shifted))))); plan[cat] = (neg, pos, keep)
    values = []
    for seed in range(2026, 2126):
        rng = np.random.default_rng(seed); aps = []
        for cat in categories:
            neg, pos, keep = plan[cat]; chosen = np.concatenate([neg, rng.choice(pos, size=keep, replace=False)]); aps.append(float(average_precision_score(y[chosen], scores[chosen])))
        values.append(float(np.mean(aps)))
    arr = np.asarray(values); return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)), "p05": float(np.quantile(arr, .05)), "p50": float(np.quantile(arr, .50)), "p95": float(np.quantile(arr, .95))}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--matches", type=Path, required=True); p.add_argument("--items", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    matches = pd.read_parquet(a.matches, columns=["id1", "id2", "target"]); manifest = load_frozen_split_manifest(); validate_frozen_split_against_matches(matches, manifest)
    rows, fold_ids = [], []
    for fold, fold_rows in enumerate(manifest["fold_rows"]): rows.extend(fold_rows); fold_ids.extend([fold] * len(fold_rows))
    dev = matches.iloc[rows].reset_index(drop=True); folds = np.asarray(fold_ids, dtype=np.int8); wanted = pd.unique(pd.concat([dev.id1, dev.id2], ignore_index=True)); items = select_items_by_ids(a.items, wanted, include_attributes=True); load_seconds = time.perf_counter() - started
    t = time.perf_counter(); features = build_sparse_pair_features(items, dev[["id1", "id2"]]); sparse_seconds = time.perf_counter() - t
    pair_frame = dev[["id1", "id2", "target"]].copy(); pair_frame["category"] = features["category"].astype(str).to_numpy()
    t = time.perf_counter(); overlap = crossfit_weighted_overlap(items, pair_frame, folds, n_features=65536, min_category_rows=1200, local_blend=0.40); overlap_seconds = time.perf_counter() - t; features["weighted_overlap"] = overlap
    overlap_ap = float(macro_ap_report(pair_frame, overlap, strict_official=True)["macro_average_precision"]); print(f"[v11-weighted] rows={len(dev):,} sparse={sparse_seconds:.3f}s overlap={overlap_seconds:.3f}s overlap_ap={overlap_ap:.12f}", flush=True)
    t = time.perf_counter(); base = crossfit_hgb_scores(features, dev.target.to_numpy(np.int8), folds, min_local_rows=1200, local_blend=0.35); hgb_seconds = time.perf_counter() - t; base_ap = float(macro_ap_report(pair_frame, base, strict_official=True)["macro_average_precision"])
    graph = np.empty_like(base); folds_out = []
    for fold in sorted(np.unique(folds).tolist()):
        mask = folds == fold; fw = pair_frame.loc[mask].reset_index(drop=True); bs = base[mask]; gf = graph_features(fw[["id1", "id2", "category"]], bs); gs = graph_rescore(bs, gf, **GRAPH_CONFIG); graph[mask] = gs
        b = float(macro_ap_report(fw, bs)["macro_average_precision"]); g = float(macro_ap_report(fw, gs)["macro_average_precision"]); folds_out.append({"fold": int(fold), "base": b, "graph": g, "delta": g-b})
    graph_ap = float(macro_ap_report(pair_frame, graph, strict_official=True)["macro_average_precision"])
    summary = {"version":"v11-weighted-overlap-1","rows":len(dev),"items":len(items),"overlap_oof":overlap_ap,"strict_oof":base_ap,"graph_oof":graph_ap,"graph_delta":graph_ap-base_ap,"stress_overlap":stress(pair_frame,overlap),"stress_base":stress(pair_frame,base),"stress_graph":stress(pair_frame,graph),"folds":folds_out,"prevalence_ratio":PREVALENCE_RATIO,"timing_seconds":{"load":load_seconds,"sparse_features":sparse_seconds,"weighted_overlap_crossfit":overlap_seconds,"hgb_crossfit":hgb_seconds,"total":time.perf_counter()-started},"sealed_gold_evaluated":False}
    print("V11_WEIGHTED_OOF="+json.dumps(summary,ensure_ascii=False,sort_keys=True),flush=True); pd.DataFrame({"row_index":np.asarray(rows,np.int64),"fold":folds,"overlap":overlap,"score":base,"graph_score":graph}).to_parquet(a.output_dir/"v11-weighted-oof.parquet",index=False); (a.output_dir/"weighted-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")


if __name__ == "__main__": main()
