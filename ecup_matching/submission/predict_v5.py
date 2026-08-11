from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

from ecup_matching.ml.data_subset import select_items_by_ids
from ecup_matching.ml.features import normalize_items
from ecup_matching.ml.features_v2 import build_pair_features_v2
from ecup_matching.ml.v5_category_specialists import predict_category_specialists
from ecup_matching.ml.v5_explicit_attributes import (
    build_explicit_attribute_features,
    build_explicit_leaf_cache,
)
from ecup_matching.ml.v5_production import category_shrunk_hgb_equal_rank_fusion


def _load_legacy_modules(root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return (
        importlib.import_module("legacy_ecup.ml.features"),
        importlib.import_module("legacy_ecup.ml.features_v2"),
        importlib.import_module("legacy_ecup.ml.textnorm"),
        importlib.import_module("legacy_ecup.ml.v5_item_text"),
        importlib.import_module("legacy_ecup.ml.v5_sparse"),
    )


def _device_and_batch(*, preferred_cuda_batch: int, cpu_batch: int) -> tuple[str, int]:
    try:
        import torch
    except ImportError:
        return "cpu", int(cpu_batch)
    if torch.cuda.is_available():
        try:
            total = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            total = 0
        if total >= 60 * 1024**3:
            return "cuda", int(preferred_cuda_batch)
        if total >= 20 * 1024**3:
            return "cuda", max(32, int(preferred_cuda_batch // 2))
        return "cuda", max(16, int(preferred_cuda_batch // 4))
    return "cpu", int(cpu_batch)


def _mean_pool(last_hidden_state, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return torch.nn.functional.normalize(summed / denom, p=2, dim=1)


def _legacy_text_cache(
    items: pd.DataFrame,
    legacy_textnorm,
    legacy_item_text,
    *,
    teacher: bool,
    norm_cache: dict[object, object] | None = None,
) -> dict[object, str]:
    """Serialize items for one neural model.

    The contrastive and teacher caches differ only in ``max_chars`` and the
    category prefix; ``normalize_item`` returns the same ``ItemNorm`` for both.
    Passing a shared ``norm_cache`` across the two calls avoids normalizing
    every item a second time, which is pure Python and scales with item count.
    """
    result: dict[object, str] = {}
    max_chars = 850 if teacher else 700
    for item_id, name, attributes, category in items[
        ["id", "name", "attributes", "category"]
    ].itertuples(index=False, name=None):
        norm = None if norm_cache is None else norm_cache.get(item_id)
        if norm is None:
            norm = legacy_textnorm.normalize_item(item_id, name, attributes, category)
            if norm_cache is not None:
                norm_cache[item_id] = norm
        body = legacy_item_text.serialize_item_v5(norm, max_chars=max_chars)
        result[item_id] = f"[CAT] {norm.category}\n{body}" if teacher else body
    return result


def _contrastive_scores(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    model_dir: Path,
    legacy_textnorm,
    legacy_item_text,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device, batch_size = _device_and_batch(preferred_cuda_batch=768, cpu_batch=32)
    texts = _legacy_text_cache(items, legacy_textnorm, legacy_item_text, teacher=False)
    unique_ids = pd.unique(pd.concat([pairs["id1"], pairs["id2"]], ignore_index=True))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device)
    model.eval()
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(unique_ids), batch_size):
            ids = unique_ids[start : start + batch_size]
            tokens = tokenizer(
                [texts[x] for x in ids],
                padding=True,
                truncation=True,
                max_length=96,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            hidden = model(**tokens).last_hidden_state
            emb = _mean_pool(hidden, tokens["attention_mask"])
            embeddings.append(emb.detach().cpu().numpy().astype(np.float32))
            if (start // batch_size) % 20 == 0:
                print(f"[v5] contrastive items {min(start + len(ids), len(unique_ids)):,}/{len(unique_ids):,}", flush=True)
    matrix = np.concatenate(embeddings, axis=0)
    index = {item_id: row for row, item_id in enumerate(unique_ids.tolist())}
    left = np.fromiter((index[x] for x in pairs["id1"].tolist()), dtype=np.int64, count=len(pairs))
    right = np.fromiter((index[x] for x in pairs["id2"].tolist()), dtype=np.int64, count=len(pairs))
    score = np.einsum("ij,ij->i", matrix[left], matrix[right], optimize=True).astype(np.float64)
    del model, tokenizer, matrix, embeddings
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return score


def _teacher_scores(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    model_dir: Path,
    legacy_textnorm,
    legacy_item_text,
) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device, batch_size = _device_and_batch(preferred_cuda_batch=256, cpu_batch=16)
    texts = _legacy_text_cache(items, legacy_textnorm, legacy_item_text, teacher=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True).to(device)
    model.eval()
    pieces: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs.iloc[start : start + batch_size]
            tokens = tokenizer(
                [texts[x] for x in chunk["id1"]],
                [texts[x] for x in chunk["id2"]],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            score = torch.sigmoid(model(**tokens).logits.squeeze(-1))
            pieces.append(score.detach().cpu().numpy().astype(np.float64))
            if (start // batch_size) % 25 == 0:
                print(f"[v5] teacher pairs {min(start + len(chunk), len(pairs)):,}/{len(pairs):,}", flush=True)
    result = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float64)
    del model, tokenizer, pieces
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def _explicit_scores(
    *,
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    base_features: pd.DataFrame,
    item_cache,
    bundle: dict,
    canonical_values: bool,
) -> np.ndarray:
    raw_categories = pairs["category"].astype(str).to_numpy()
    leaf_cache = build_explicit_leaf_cache(item_cache, canonical_values=canonical_values)
    score = np.full(len(pairs), np.nan, dtype=np.float64)
    for category in sorted(np.unique(raw_categories).tolist()):
        positions = np.flatnonzero(raw_categories == category)
        if category not in bundle["models"]:
            raise ValueError(f"explicit production model missing category {category!r}")
        category_pairs = pairs.iloc[positions].reset_index(drop=True)
        attr = build_explicit_attribute_features(
            items,
            category_pairs,
            bundle["key_spec"],
            item_cache=item_cache,
            category=category,
            leaf_cache=leaf_cache,
        )
        base = base_features.iloc[positions].drop(columns=["category"]).reset_index(drop=True)
        x = pd.concat([base, attr.reset_index(drop=True)], axis=1).to_numpy(dtype=np.float32)
        score[positions] = bundle["models"][category].predict_proba(x)[:, 1]
    if not np.isfinite(score).all():
        raise RuntimeError("explicit model failed to score every pair")
    return score


def predict_to_csv_v5(
    *,
    items_path: Path,
    matches_path: Path,
    structured_model_path: Path,
    contrastive_model_dir: Path,
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

    print(f"[v5] pairs={len(pairs):,} items={len(items):,}", flush=True)
    legacy_base = legacy_features_v2.build_features_v2_chunked(
        items, pairs, attribute_importance=None, chunk_size=25_000
    )
    weak = predict_category_specialists(structured["weak"], legacy_base)

    sparse_extra = legacy_sparse.transform_sparse_pairs(structured["sparse"]["encoder"], items, pairs)
    sparse_features = pd.concat([legacy_base.reset_index(drop=True), sparse_extra.reset_index(drop=True)], axis=1)
    sparse = predict_category_specialists(structured["sparse"]["specialists"], sparse_features)
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

    contrastive = _contrastive_scores(
        items, pairs, contrastive_model_dir, legacy_textnorm, legacy_item_text
    )
    teacher = _teacher_scores(
        items, pairs, teacher_model_dir, legacy_textnorm, legacy_item_text
    )

    final = category_shrunk_hgb_equal_rank_fusion(
        {
            "weak": weak,
            "sparse": sparse,
            "explicit": explicit,
            "contrastive": contrastive,
            "teacher": teacher,
            "typed_explicit": typed_explicit,
        },
        pairs["category"].astype(str).to_numpy(),
        category_model,
        hgb_bundle,
    )
    if len(final) != len(pairs) or not np.isfinite(final).all():
        raise RuntimeError("v5 final score is incomplete or non-finite")
    result = pairs[["id1", "id2"]].copy()
    result["predict"] = np.clip(final, 0.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"[v5] wrote {output_path} rows={len(result):,} total_seconds={time.perf_counter()-started:.2f}", flush=True)
    return result
