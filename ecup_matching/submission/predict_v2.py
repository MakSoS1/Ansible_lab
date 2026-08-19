from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecup_matching.ml.features import normalize_items
from ecup_matching.ml.features_v2 import FEATURE_NAMES_V2, build_pair_features_v2
from ecup_matching.ml.model_io import load_model_bundle


def predict_to_csv_v2(
    items_path: Path,
    matches_path: Path,
    model_path: Path,
    manifest_path: Path,
    output_path: Path,
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    total_started = time.perf_counter()
    load_started = time.perf_counter()
    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(matches_path, columns=["id1", "id2"])
    model, manifest = load_model_bundle(model_path, manifest_path)

    expected = manifest.get("feature_names")
    if list(expected or []) != list(FEATURE_NAMES_V2):
        raise RuntimeError("v2 model feature manifest does not match runtime FEATURE_NAMES_V2")
    importance = manifest.get("attribute_importance")
    if not isinstance(importance, dict):
        raise RuntimeError("v2 manifest is missing attribute_importance")
    item_cache = normalize_items(items)
    print(
        f"[v2] loaded+normalized {len(items):,} items and {len(matches):,} pairs "
        f"in {time.perf_counter()-load_started:.2f}s",
        flush=True,
    )

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    scores_parts: list[np.ndarray] = []
    feature_seconds = 0.0
    predict_seconds = 0.0
    for start in range(0, len(matches), chunk_size):
        pair_chunk = matches.iloc[start : start + chunk_size]
        feat_started = time.perf_counter()
        features = build_pair_features_v2(
            items,
            pair_chunk,
            attribute_importance=importance,
            item_cache=item_cache,
        )
        feature_seconds += time.perf_counter() - feat_started
        pred_started = time.perf_counter()
        chunk_scores = model.predict_proba(features)[:, 1]
        predict_seconds += time.perf_counter() - pred_started
        if not np.isfinite(chunk_scores).all():
            raise RuntimeError("v2 prediction contains NaN or infinity")
        scores_parts.append(np.clip(chunk_scores.astype(np.float64), 0.0, 1.0))
        print(
            f"[v2] processed {min(start + len(pair_chunk), len(matches)):,}/{len(matches):,} pairs",
            flush=True,
        )

    scores = np.concatenate(scores_parts) if scores_parts else np.empty(0, dtype=np.float64)
    print(
        f"[v2] feature_seconds={feature_seconds:.2f} predict_seconds={predict_seconds:.2f}",
        flush=True,
    )
    result = matches[["id1", "id2"]].copy()
    result["predict"] = scores
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(
        f"[v2] wrote {output_path} total={time.perf_counter()-total_started:.2f}s",
        flush=True,
    )
    return result
