from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecup_matching.ml.batch_features import build_features_chunked
from ecup_matching.ml.features import FEATURE_NAMES
from ecup_matching.ml.model_io import load_model_bundle


def predict_to_csv(
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
    if expected is not None and list(expected) != list(FEATURE_NAMES):
        raise RuntimeError("model feature manifest does not match runtime FEATURE_NAMES")
    print(f"[v1] loaded {len(items):,} items and {len(matches):,} pairs in {time.perf_counter()-load_started:.2f}s")

    feat_started = time.perf_counter()
    features = build_features_chunked(items, matches, chunk_size=chunk_size)
    print(f"[v1] built {len(features):,} feature rows in {time.perf_counter()-feat_started:.2f}s")

    pred_started = time.perf_counter()
    scores = model.predict_proba(features)[:, 1]
    if not np.isfinite(scores).all():
        raise RuntimeError("prediction contains NaN or infinity")
    scores = np.clip(scores.astype(np.float64), 0.0, 1.0)
    print(f"[v1] predicted {len(scores):,} rows in {time.perf_counter()-pred_started:.2f}s")

    result = matches[["id1", "id2"]].copy()
    result["predict"] = scores
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"[v1] wrote {output_path} total={time.perf_counter()-total_started:.2f}s")
    return result
