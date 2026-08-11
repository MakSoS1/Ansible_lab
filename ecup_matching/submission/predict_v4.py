from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecup_matching.ml.features import normalize_items
from ecup_matching.ml.features_v2 import FEATURE_NAMES_V2, build_pair_features_v2
from ecup_matching.ml.model_io import load_model_bundle
from ecup_matching.submission.predict_v3 import (
    _pair_categories,
    _predict_neural_subset,
    apply_category_blend,
    categories_requiring_neural,
)


def choose_neural_batch_size(
    requested: int,
    *,
    cuda_available: bool,
    cuda_memory_bytes: int | None,
) -> int:
    if requested <= 0:
        raise ValueError("requested neural batch size must be positive")
    if not cuda_available:
        return min(requested, 16)
    if cuda_memory_bytes is None or cuda_memory_bytes <= 0:
        return min(requested, 32)
    gib = float(cuda_memory_bytes) / float(1024**3)
    if gib < 12.0:
        cap = 32
    elif gib < 32.0:
        cap = 128
    else:
        cap = 512
    return min(requested, cap)


def _runtime_neural_batch_size(requested: int) -> int:
    try:
        import torch
    except ImportError:
        return choose_neural_batch_size(
            requested,
            cuda_available=False,
            cuda_memory_bytes=None,
        )
    cuda_available = bool(torch.cuda.is_available())
    memory: int | None = None
    if cuda_available:
        try:
            memory = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            memory = None
    return choose_neural_batch_size(
        requested,
        cuda_available=cuda_available,
        cuda_memory_bytes=memory,
    )


def predict_to_csv_v4(
    items_path: Path,
    matches_path: Path,
    structured_model_path: Path,
    structured_manifest_path: Path,
    neural_model_dir: Path,
    neural_manifest_path: Path,
    output_path: Path,
    *,
    structured_chunk_size: int = 50_000,
    neural_batch_size: int = 512,
) -> pd.DataFrame:
    total_started = time.perf_counter()
    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(matches_path, columns=["id1", "id2"])
    structured_model, structured_manifest = load_model_bundle(
        structured_model_path, structured_manifest_path
    )
    expected = structured_manifest.get("feature_names")
    if list(expected or []) != list(FEATURE_NAMES_V2):
        raise RuntimeError("v4 structured manifest does not match runtime FEATURE_NAMES_V2")
    importance = structured_manifest.get("attribute_importance")
    if not isinstance(importance, dict):
        raise RuntimeError("v4 structured manifest is missing attribute_importance")
    neural_manifest = json.loads(Path(neural_manifest_path).read_text(encoding="utf-8"))
    if neural_manifest.get("version") != "v4-strong-reranker":
        raise RuntimeError("unexpected v4 neural manifest version")
    if neural_manifest.get("base_model") != "ai-forever/ruBert-base":
        raise RuntimeError("unexpected v4 base model")
    revision = str(neural_manifest.get("base_model_revision", ""))
    if len(revision) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in revision):
        raise RuntimeError("v4 manifest is missing an exact base-model revision")

    item_cache = normalize_items(items)
    categories = _pair_categories(matches, item_cache)
    required_categories = categories_requiring_neural(neural_manifest)

    if structured_chunk_size <= 0:
        raise ValueError("structured_chunk_size must be positive")
    structured_parts: list[np.ndarray] = []
    feature_seconds = 0.0
    for start in range(0, len(matches), structured_chunk_size):
        chunk = matches.iloc[start : start + structured_chunk_size]
        feat_started = time.perf_counter()
        features = build_pair_features_v2(
            items,
            chunk,
            attribute_importance=importance,
            item_cache=item_cache,
        )
        feature_seconds += time.perf_counter() - feat_started
        scores = structured_model.predict_proba(features)[:, 1]
        if not np.isfinite(scores).all():
            raise RuntimeError("v4 structured prediction contains NaN or infinity")
        structured_parts.append(np.clip(scores.astype(np.float64), 0.0, 1.0))
        print(
            f"[v4] structured {min(start + len(chunk), len(matches)):,}/{len(matches):,}",
            flush=True,
        )
    structured = (
        np.concatenate(structured_parts)
        if structured_parts
        else np.empty(0, dtype=np.float64)
    )

    if "*" in required_categories:
        candidate_mask = np.ones(len(categories), dtype=bool)
    else:
        candidate_mask = np.isin(
            categories, np.asarray(sorted(required_categories), dtype=object)
        )
    candidate_positions = np.flatnonzero(candidate_mask)
    neural = structured.copy()
    neural_seconds = 0.0
    if len(candidate_positions):
        effective_batch_size = _runtime_neural_batch_size(neural_batch_size)
        print(
            f"[v4] neural_batch_size={effective_batch_size} requested={neural_batch_size}",
            flush=True,
        )
        neural_started = time.perf_counter()
        candidate_scores = _predict_neural_subset(
            matches=matches,
            candidate_positions=candidate_positions,
            item_cache=item_cache,
            attribute_importance=importance,
            model_dir=neural_model_dir,
            manifest=neural_manifest,
            batch_size=effective_batch_size,
        )
        neural[candidate_positions] = candidate_scores
        neural_seconds = time.perf_counter() - neural_started

    final_scores = apply_category_blend(
        categories, structured, neural, neural_manifest
    )
    if len(final_scores) != len(matches) or not np.isfinite(final_scores).all():
        raise RuntimeError("v4 final prediction is incomplete or non-finite")
    result = matches[["id1", "id2"]].copy()
    result["predict"] = np.clip(final_scores, 0.0, 1.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(
        f"[v4] wrote {output_path} pairs={len(matches):,} neural_pairs={len(candidate_positions):,} "
        f"feature_seconds={feature_seconds:.2f} neural_seconds={neural_seconds:.2f} "
        f"total={time.perf_counter()-total_started:.2f}s",
        flush=True,
    )
    return result
