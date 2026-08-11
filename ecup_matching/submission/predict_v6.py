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
from ecup_matching.ml.v6_teacher_gate import (
    assemble_partial_teacher_signal,
    disagreement_gate_mask,
)
from ecup_matching.submission.predict_v5 import (
    _explicit_scores,
    _legacy_text_cache,
    _load_legacy_modules,
    _mean_pool,
)
from ecup_matching.submission.v6_fast import select_runtime_config


def _phase(label: str, started: float, previous: float) -> float:
    now = time.perf_counter()
    print(
        f"[v6] phase={label} seconds={now-previous:.3f} total_seconds={now-started:.3f}",
        flush=True,
    )
    return now


def _runtime_config():
    try:
        import torch
    except ImportError:
        return select_runtime_config(total_memory_bytes=0, device_type="cpu")
    if not torch.cuda.is_available():
        return select_runtime_config(total_memory_bytes=0, device_type="cpu")
    try:
        total = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        total = 0
    return select_runtime_config(total_memory_bytes=total, device_type="cuda")


def _load_auto_model(model_class, model_dir: Path, *, device: str):
    kwargs = {"local_files_only": True}
    if device == "cuda":
        try:
            return model_class.from_pretrained(
                str(model_dir), attn_implementation="sdpa", **kwargs
            ).to(device)
        except (TypeError, ValueError, NotImplementedError):
            pass
    return model_class.from_pretrained(str(model_dir), **kwargs).to(device)


def _contrastive_scores_fast(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    model_dir: Path,
    legacy_textnorm,
    legacy_item_text,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    config = _runtime_config()
    device = config.device
    active_batch = int(config.contrastive_batch)
    texts = _legacy_text_cache(items, legacy_textnorm, legacy_item_text, teacher=False)
    unique_ids = pd.unique(pd.concat([pairs["id1"], pairs["id2"]], ignore_index=True)).tolist()
    ordered_ids = sorted(unique_ids, key=lambda item_id: (len(texts[item_id]), str(item_id)))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = _load_auto_model(AutoModel, model_dir, device=device)
    model.eval()

    embedding_by_id: dict[object, np.ndarray] = {}
    position = 0
    with torch.inference_mode():
        while position < len(ordered_ids):
            ids = ordered_ids[position : position + active_batch]
            try:
                tokens = tokenizer(
                    [texts[item_id] for item_id in ids],
                    padding=True,
                    truncation=True,
                    max_length=96,
                    return_tensors="pt",
                )
                tokens = {
                    key: value.to(device, non_blocking=(device == "cuda"))
                    for key, value in tokens.items()
                }
                hidden = model(**tokens).last_hidden_state
                emb = _mean_pool(hidden, tokens["attention_mask"])
                array = emb.detach().cpu().numpy().astype(np.float32, copy=False)
            except torch.OutOfMemoryError:
                if device != "cuda" or active_batch <= 16:
                    raise
                torch.cuda.empty_cache()
                active_batch = max(16, active_batch // 2)
                print(f"[v6] contrastive OOM fallback batch={active_batch}", flush=True)
                continue
            for item_id, vector in zip(ids, array, strict=True):
                embedding_by_id[item_id] = vector
            position += len(ids)
            if position <= active_batch or position % max(active_batch * 20, 1) < active_batch:
                print(
                    f"[v6] contrastive items {position:,}/{len(ordered_ids):,} batch={active_batch}",
                    flush=True,
                )

    left_ids = pairs["id1"].to_numpy()
    right_ids = pairs["id2"].to_numpy()
    left = np.stack([embedding_by_id[item_id] for item_id in left_ids])
    right = np.stack([embedding_by_id[item_id] for item_id in right_ids])
    score = np.einsum("ij,ij->i", left, right, optimize=True).astype(np.float64)
    del model, tokenizer, embedding_by_id, left, right
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return score


def _teacher_selected_scores_fast(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    selected_indices: np.ndarray,
    model_dir: Path,
    legacy_textnorm,
    legacy_item_text,
) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if selected_indices.ndim != 1:
        raise ValueError("selected_indices must be one-dimensional")
    if len(selected_indices) == 0:
        return np.empty(0, dtype=np.float64)

    config = _runtime_config()
    device = config.device
    active_batch = int(config.teacher_batch)
    texts = _legacy_text_cache(items, legacy_textnorm, legacy_item_text, teacher=True)
    left_all = pairs["id1"].to_numpy()
    right_all = pairs["id2"].to_numpy()
    selected_left = left_all[selected_indices]
    selected_right = right_all[selected_indices]
    order = np.asarray(
        sorted(
            range(len(selected_indices)),
            key=lambda i: (
                max(len(texts[selected_left[i]]), len(texts[selected_right[i]])),
                i,
            ),
        ),
        dtype=np.int64,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = _load_auto_model(
        AutoModelForSequenceClassification, model_dir, device=device
    )
    model.eval()
    result = np.empty(len(selected_indices), dtype=np.float64)
    position = 0
    with torch.inference_mode():
        while position < len(order):
            local = order[position : position + active_batch]
            try:
                tokens = tokenizer(
                    [texts[selected_left[i]] for i in local],
                    [texts[selected_right[i]] for i in local],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                tokens = {
                    key: value.to(device, non_blocking=(device == "cuda"))
                    for key, value in tokens.items()
                }
                values = torch.sigmoid(model(**tokens).logits.squeeze(-1))
                array = values.detach().cpu().numpy().astype(np.float64, copy=False)
            except torch.OutOfMemoryError:
                if device != "cuda" or active_batch <= 8:
                    raise
                torch.cuda.empty_cache()
                active_batch = max(8, active_batch // 2)
                print(f"[v6] teacher OOM fallback batch={active_batch}", flush=True)
                continue
            result[local] = array
            position += len(local)
            if position <= active_batch or position % max(active_batch * 25, 1) < active_batch:
                print(
                    f"[v6] teacher selected {position:,}/{len(order):,} batch={active_batch}",
                    flush=True,
                )
    del model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def predict_to_csv_v6(
    *,
    coverage: float,
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
    config = _runtime_config()
    print(
        f"[v6] pairs={len(pairs):,} items={len(items):,} coverage={coverage:.2f} "
        f"device={config.device} contrastive_batch={config.contrastive_batch} "
        f"teacher_batch={config.teacher_batch}",
        flush=True,
    )
    previous = _phase("load", started, previous)

    legacy_base = legacy_features_v2.build_features_v2_chunked(
        items, pairs, attribute_importance=None, chunk_size=50_000
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

    contrastive = _contrastive_scores_fast(
        items, pairs, contrastive_model_dir, legacy_textnorm, legacy_item_text
    )
    previous = _phase("contrastive", started, previous)

    non_teacher = {
        "weak": weak,
        "sparse": sparse,
        "explicit": explicit,
        "contrastive": contrastive,
        "typed_explicit": typed_explicit,
    }
    categories = pairs["category"].astype(str).to_numpy()
    gate_mask = disagreement_gate_mask(non_teacher, categories, coverage=coverage)
    selected_indices = np.flatnonzero(gate_mask)
    print(
        f"[v6] teacher_gate selected={len(selected_indices):,}/{len(pairs):,} "
        f"fraction={gate_mask.mean():.6f}",
        flush=True,
    )
    previous = _phase("gate", started, previous)

    selected_teacher = _teacher_selected_scores_fast(
        items,
        pairs,
        selected_indices,
        teacher_model_dir,
        legacy_textnorm,
        legacy_item_text,
    )
    teacher_signal, verified_mask = assemble_partial_teacher_signal(
        non_teacher,
        categories,
        coverage=coverage,
        selected_teacher_scores=selected_teacher,
    )
    if not np.array_equal(gate_mask, verified_mask):
        raise RuntimeError("teacher gate changed between selection and assembly")
    previous = _phase("teacher", started, previous)

    final = category_shrunk_hgb_equal_rank_fusion(
        {**non_teacher, "teacher": teacher_signal},
        categories,
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
        f"[v6] wrote {output_path} rows={len(result):,} "
        f"total_seconds={time.perf_counter()-started:.2f}",
        flush=True,
    )
    return result
