from __future__ import annotations

from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd

from ecup_matching.ml.data_subset import select_items_by_ids
from ecup_matching.ml.v5_production import category_shrunk_hgb_equal_rank_fusion
from ecup_matching.ml.v6_fast_ablation import build_fast_candidate_scores
from ecup_matching.ml.v8_graph import graph_features, graph_rescore
from ecup_matching.submission.predict_v5 import _load_legacy_modules
from ecup_matching.submission.predict_v6 import _structured_scores_streaming, _teacher_selected_scores_fast
from ecup_matching.submission.v6_parallel import resolve_worker_count

GRAPH_CONFIG = dict(
    reciprocal_best_bonus=0.0,
    reciprocal_top3_bonus=0.0,
    endpoint_rank_weight=0.02,
    ambiguity_penalty=0.01,
)


def predict_to_csv_v11_no_contrastive(
    *,
    items_path: Path,
    matches_path: Path,
    structured_model_path: Path,
    teacher_model_dir: Path,
    category_model_path: Path,
    hgb_model_path: Path,
    runtime_root: Path,
    output_path: Path,
) -> pd.DataFrame:
    started = time.perf_counter()
    pairs = pd.read_parquet(matches_path, columns=["id1", "id2"])
    needed_ids = pd.unique(pd.concat([pairs["id1"], pairs["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, needed_ids, include_attributes=True)
    raw_category = items.set_index("id")["category"].astype(str)
    pairs = pairs.copy()
    pairs["category"] = pairs["id1"].map(raw_category)
    if pairs["category"].isna().any():
        raise RuntimeError("failed to attach pair category")

    structured = joblib.load(structured_model_path)
    category_model = json.loads(category_model_path.read_text(encoding="utf-8"))
    hgb_bundle = joblib.load(hgb_model_path)
    legacy_features, legacy_features_v2, legacy_textnorm, legacy_item_text, legacy_sparse = _load_legacy_modules(runtime_root)

    scores = _structured_scores_streaming(
        items=items,
        pairs=pairs,
        structured=structured,
        legacy_features=legacy_features,
        legacy_features_v2=legacy_features_v2,
        legacy_sparse=legacy_sparse,
        workers=resolve_worker_count(),
    )
    teacher = _teacher_selected_scores_fast(
        items,
        pairs,
        np.arange(len(pairs), dtype=np.int64),
        teacher_model_dir,
        legacy_textnorm,
        legacy_item_text,
        norm_cache={},
    )
    six = {
        "weak": scores["weak"],
        "sparse": scores["sparse"],
        "explicit": scores["explicit"],
        "teacher": teacher,
        "typed_explicit": scores["typed_explicit"],
    }
    candidate = build_fast_candidate_scores(six, "no_contrastive")
    categories = pairs["category"].astype(str).to_numpy()
    base = category_shrunk_hgb_equal_rank_fusion(candidate, categories, category_model, hgb_bundle)
    graph = graph_rescore(base, graph_features(pairs[["id1", "id2", "category"]], base), **GRAPH_CONFIG)
    if len(graph) != len(pairs) or not np.isfinite(graph).all():
        raise RuntimeError("v11 produced invalid scores")
    result = pairs[["id1", "id2"]].copy()
    result["predict"] = graph
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"[v11] rows={len(result):,} seconds={time.perf_counter()-started:.2f}", flush=True)
    del scores, teacher, candidate, base, graph
    gc.collect()
    return result
