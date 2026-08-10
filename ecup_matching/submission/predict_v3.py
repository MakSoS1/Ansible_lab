from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ecup_matching.ml.features import normalize_items
from ecup_matching.ml.features_v2 import FEATURE_NAMES_V2, build_pair_features_v2
from ecup_matching.ml.model_io import load_model_bundle
from ecup_matching.ml.reranker_data import serialize_pair


def categories_requiring_neural(manifest: Mapping[str, object]) -> set[str]:
    raw = manifest.get("category_alphas")
    if not isinstance(raw, Mapping):
        raise RuntimeError("v3 manifest is missing category_alphas")
    global_alpha = float(raw.get("__global__", 0.0))
    if global_alpha > 0.0:
        return {"*"}
    return {
        str(category)
        for category, alpha in raw.items()
        if str(category) != "__global__" and float(alpha) > 0.0
    }


def apply_category_blend(
    categories,
    structured_scores,
    neural_scores,
    manifest: Mapping[str, object],
) -> np.ndarray:
    category = np.asarray(categories).astype(str)
    structured = np.asarray(structured_scores, dtype=np.float64)
    neural = np.asarray(neural_scores, dtype=np.float64)
    if not (len(category) == len(structured) == len(neural)):
        raise ValueError("category, structured and neural arrays must have equal length")
    raw = manifest.get("category_alphas")
    if not isinstance(raw, Mapping):
        raise RuntimeError("v3 manifest is missing category_alphas")
    global_alpha = float(raw.get("__global__", 0.0))
    if not 0.0 <= global_alpha <= 1.0:
        raise RuntimeError(f"invalid global neural alpha: {global_alpha}")
    out = structured.copy()
    for name in np.unique(category):
        alpha = float(raw.get(str(name), global_alpha))
        if not 0.0 <= alpha <= 1.0:
            raise RuntimeError(f"invalid neural alpha for category {name!r}: {alpha}")
        if alpha <= 0.0:
            continue
        mask = category == name
        out[mask] = (1.0 - alpha) * structured[mask] + alpha * neural[mask]
    if not np.isfinite(out).all():
        raise RuntimeError("v3 blended prediction contains NaN or infinity")
    return np.clip(out, 0.0, 1.0)


def _pair_categories(matches: pd.DataFrame, item_cache) -> np.ndarray:
    categories: list[str] = []
    for row in matches.itertuples(index=False):
        left = item_cache.get(row.id1)
        right = item_cache.get(row.id2)
        if left is None or right is None:
            raise KeyError(f"pair references missing item: {row.id1!r}, {row.id2!r}")
        categories.append(str(left.category or right.category or ""))
    return np.asarray(categories, dtype=object)


def _predict_neural_subset(
    *,
    matches: pd.DataFrame,
    candidate_positions: np.ndarray,
    item_cache,
    attribute_importance,
    model_dir: Path,
    manifest: Mapping[str, object],
    batch_size: int,
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("v3 neural runtime requires torch and transformers") from exc

    if batch_size <= 0:
        raise ValueError("neural_batch_size must be positive")
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    max_length = int(manifest.get("max_length", 160))
    max_attrs = int(manifest.get("max_attrs", 10))
    max_chars = int(manifest.get("max_chars", 700))
    print(f"[v3] neural device={device.type} rows={len(candidate_positions):,} max_length={max_length}", flush=True)

    out = np.empty(len(candidate_positions), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(candidate_positions), batch_size):
            positions = candidate_positions[start : start + batch_size]
            text_a: list[str] = []
            text_b: list[str] = []
            for pos in positions:
                row = matches.iloc[int(pos)]
                left = item_cache[row.id1]
                right = item_cache[row.id2]
                a, b = serialize_pair(
                    left,
                    right,
                    attribute_importance,
                    max_attrs=max_attrs,
                    max_chars=max_chars,
                )
                text_a.append(a)
                text_b.append(b)
            encoded = tokenizer(
                text_a,
                text_b,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(**encoded).logits.squeeze(-1)
            else:
                logits = model(**encoded).logits.squeeze(-1)
            scores = torch.sigmoid(logits).float().cpu().numpy()
            out[start : start + len(positions)] = scores
            if start == 0 or start + len(positions) == len(candidate_positions) or start % (batch_size * 20) == 0:
                print(f"[v3] neural {min(start + len(positions), len(candidate_positions)):,}/{len(candidate_positions):,}", flush=True)
    if not np.isfinite(out).all():
        raise RuntimeError("v3 neural prediction contains NaN or infinity")
    return np.clip(out.astype(np.float64), 0.0, 1.0)


def predict_to_csv_v3(
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
    structured_model, structured_manifest = load_model_bundle(structured_model_path, structured_manifest_path)
    expected = structured_manifest.get("feature_names")
    if list(expected or []) != list(FEATURE_NAMES_V2):
        raise RuntimeError("v3 structured manifest does not match runtime FEATURE_NAMES_V2")
    importance = structured_manifest.get("attribute_importance")
    if not isinstance(importance, dict):
        raise RuntimeError("v3 structured manifest is missing attribute_importance")
    neural_manifest = json.loads(Path(neural_manifest_path).read_text(encoding="utf-8"))
    if neural_manifest.get("version") != "v3-compact-reranker":
        raise RuntimeError("unexpected v3 neural manifest version")
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
        features = build_pair_features_v2(items, chunk, attribute_importance=importance, item_cache=item_cache)
        feature_seconds += time.perf_counter() - feat_started
        scores = structured_model.predict_proba(features)[:, 1]
        if not np.isfinite(scores).all():
            raise RuntimeError("v3 structured prediction contains NaN or infinity")
        structured_parts.append(np.clip(scores.astype(np.float64), 0.0, 1.0))
        print(f"[v3] structured {min(start + len(chunk), len(matches)):,}/{len(matches):,}", flush=True)
    structured = np.concatenate(structured_parts) if structured_parts else np.empty(0, dtype=np.float64)

    if "*" in required_categories:
        candidate_mask = np.ones(len(categories), dtype=bool)
    else:
        candidate_mask = np.isin(categories, np.asarray(sorted(required_categories), dtype=object))
    candidate_positions = np.flatnonzero(candidate_mask)
    neural = structured.copy()
    neural_seconds = 0.0
    if len(candidate_positions):
        neural_started = time.perf_counter()
        candidate_scores = _predict_neural_subset(
            matches=matches,
            candidate_positions=candidate_positions,
            item_cache=item_cache,
            attribute_importance=importance,
            model_dir=neural_model_dir,
            manifest=neural_manifest,
            batch_size=neural_batch_size,
        )
        neural[candidate_positions] = candidate_scores
        neural_seconds = time.perf_counter() - neural_started

    final_scores = apply_category_blend(categories, structured, neural, neural_manifest)
    result = matches[["id1", "id2"]].copy()
    result["predict"] = final_scores
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(
        f"[v3] wrote {output_path} pairs={len(matches):,} neural_pairs={len(candidate_positions):,} "
        f"feature_seconds={feature_seconds:.2f} neural_seconds={neural_seconds:.2f} total={time.perf_counter()-total_started:.2f}s",
        flush=True,
    )
    return result
