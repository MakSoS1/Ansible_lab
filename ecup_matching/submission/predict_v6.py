from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from ecup_matching.ml.data_subset import select_items_by_ids
from ecup_matching.ml.features import normalize_items
from ecup_matching.ml.features_v2 import build_pair_features_v2
from ecup_matching.ml.v5_category_specialists import predict_category_specialists
from ecup_matching.ml.v5_production import category_shrunk_hgb_equal_rank_fusion
from ecup_matching.ml.v6_fast_ablation import CANDIDATE_SPECS, build_fast_candidate_scores
from ecup_matching.submission.predict_v5 import (
    _contrastive_scores,
    _explicit_scores,
    _load_legacy_modules,
    _teacher_scores,
)


def _phase(label: str, started: float, previous: float) -> float:
    now = time.perf_counter()
    print(
        f"[v6] phase={label} seconds={now-previous:.3f} total_seconds={now-started:.3f}",
        flush=True,
    )
    return now


def predict_to_csv_v6(
    *,
    candidate: str,
    items_path: Path,
    matches_path: Path,
    structured_model_path: Path,
    contrastive_model_dir: Path | None,
    teacher_model_dir: Path | None,
    category_model_path: Path,
    hgb_model_path: Path,
    runtime_root: Path,
    output_path: Path,
) -> pd.DataFrame:
    if candidate not in CANDIDATE_SPECS:
        raise ValueError(f"unknown fast candidate: {candidate}")
    spec = CANDIDATE_SPECS[candidate]
    started = time.perf_counter()
    previous = started

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
    legacy_features, legacy_features_v2, legacy_textnorm, legacy_item_text, legacy_sparse = (
        _load_legacy_modules(runtime_root)
    )
    print(
        f"[v6] candidate={candidate} pairs={len(pairs):,} items={len(items):,} "
        f"expensive={','.join(spec.required_expensive_signals) or 'none'}",
        flush=True,
    )
    previous = _phase("load", started, previous)

    legacy_base = legacy_features_v2.build_features_v2_chunked(
        items, pairs, attribute_importance=None, chunk_size=25_000
    )
    weak = predict_category_specialists(structured["weak"], legacy_base)

    sparse_extra = legacy_sparse.transform_sparse_pairs(
        structured["sparse"]["encoder"], items, pairs
    )
    sparse_features = pd.concat(
        [legacy_base.reset_index(drop=True), sparse_extra.reset_index(drop=True)], axis=1
    )
    sparse = predict_category_specialists(
        structured["sparse"]["specialists"], sparse_features
    )
    del sparse_extra, sparse_features
    gc.collect()

    legacy_cache = legacy_features.normalize_items(items)
    explicit = _explicit_scores(
        items=items,
        pairs=pairs,
        base_features=legacy_base,
        item_cache=legacy_cache,
        bundle=structured["explicit"],
        canonical_values=False,
    )
    del legacy_cache

    typed_cache = normalize_items(items)
    typed_base = build_pair_features_v2(items, pairs, item_cache=typed_cache)
    typed_explicit = _explicit_scores(
        items=items,
        pairs=pairs,
        base_features=typed_base,
        item_cache=typed_cache,
        bundle=structured["typed_explicit"],
        canonical_values=True,
    )
    del typed_cache, typed_base, legacy_base
    gc.collect()
    previous = _phase("structured", started, previous)

    available: dict[str, np.ndarray] = {
        "weak": weak,
        "sparse": sparse,
        "explicit": explicit,
        "typed_explicit": typed_explicit,
    }
    if "contrastive" in spec.required_expensive_signals:
        if contrastive_model_dir is None or not contrastive_model_dir.is_dir():
            raise FileNotFoundError("contrastive model required by selected v6 candidate")
        available["contrastive"] = _contrastive_scores(
            items, pairs, contrastive_model_dir, legacy_textnorm, legacy_item_text
        )
        previous = _phase("contrastive", started, previous)
    if "teacher" in spec.required_expensive_signals:
        if teacher_model_dir is None or not teacher_model_dir.is_dir():
            raise FileNotFoundError("teacher model required by selected v6 candidate")
        available["teacher"] = _teacher_scores(
            items, pairs, teacher_model_dir, legacy_textnorm, legacy_item_text
        )
        previous = _phase("teacher", started, previous)

    six_scores = build_fast_candidate_scores(available, candidate)
    previous = _phase("surrogate", started, previous)
    final = category_shrunk_hgb_equal_rank_fusion(
        six_scores,
        pairs["category"].astype(str).to_numpy(),
        category_model,
        hgb_bundle,
    )
    previous = _phase("meta", started, previous)

    if len(final) != len(pairs) or not np.isfinite(final).all():
        raise RuntimeError("v6 final score is incomplete or non-finite")
    result = pairs[["id1", "id2"]].copy()
    result["predict"] = np.clip(final, 0.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    _phase("write", started, previous)
    print(
        f"[v6] wrote {output_path} rows={len(result):,} total_seconds={time.perf_counter()-started:.2f}",
        flush=True,
    )
    return result
