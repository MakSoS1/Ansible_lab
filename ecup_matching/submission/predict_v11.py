from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ecup_matching.ml.data_subset import select_items_by_ids
from ecup_matching.ml.v8_graph import graph_features, graph_rescore
from ecup_matching.ml.v11_fastlex import build_fast_pair_features
from ecup_matching.ml.v11_sparse import SparseConfig, sparse_pair_scores
from ecup_matching.ml.v11_stack import predict_hgb_bundle


GRAPH_CONFIG = dict(
    reciprocal_best_bonus=0.0,
    reciprocal_top3_bonus=0.0,
    endpoint_rank_weight=0.02,
    ambiguity_penalty=0.01,
)


def predict_to_csv_v11(
    *,
    items_path: Path,
    matches_path: Path,
    model_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    started = time.perf_counter()
    pairs = pd.read_parquet(matches_path, columns=["id1", "id2"])
    needed = pd.unique(pd.concat([pairs.id1, pairs.id2], ignore_index=True))
    items = select_items_by_ids(items_path, needed, include_attributes=True)
    print(f"[v11] load pairs={len(pairs):,} items={len(items):,} seconds={time.perf_counter()-started:.3f}", flush=True)

    t = time.perf_counter()
    features = build_fast_pair_features(items, pairs)
    print(f"[v11] fastlex seconds={time.perf_counter()-t:.3f}", flush=True)

    t = time.perf_counter()
    features["sparse_cosine"] = sparse_pair_scores(
        items, pairs, config=SparseConfig(n_features=65536)
    )
    print(f"[v11] sparse seconds={time.perf_counter()-t:.3f}", flush=True)

    t = time.perf_counter()
    bundle = joblib.load(model_path)
    base = predict_hgb_bundle(bundle, features)
    print(f"[v11] hgb seconds={time.perf_counter()-t:.3f}", flush=True)

    t = time.perf_counter()
    graph_input = pairs.copy()
    graph_input["category"] = features["category"].astype(str).to_numpy()
    gf = graph_features(graph_input[["id1", "id2", "category"]], base)
    final = graph_rescore(base, gf, **GRAPH_CONFIG)
    if len(final) != len(pairs) or not np.isfinite(final).all():
        raise RuntimeError("v11 produced incomplete or non-finite scores")
    print(f"[v11] graph seconds={time.perf_counter()-t:.3f}", flush=True)

    result = pairs.copy()
    result["predict"] = final.astype(np.float64, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"[v11] total seconds={time.perf_counter()-started:.3f}", flush=True)
    return result


__all__ = ["GRAPH_CONFIG", "predict_to_csv_v11"]
