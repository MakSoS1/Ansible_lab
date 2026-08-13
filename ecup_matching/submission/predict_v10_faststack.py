from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from ecup_matching.ml.data_subset import select_items_by_ids
from ecup_matching.ml.v5_production import category_shrunk_hgb_equal_rank_fusion
from ecup_matching.ml.v6_fast_ablation import build_fast_candidate_scores
from ecup_matching.ml.v8_graph import graph_features, graph_rescore
from ecup_matching.submission.predict_v5 import _load_legacy_modules
from ecup_matching.submission.predict_v6 import (
    STRUCTURED_CHUNK_SIZE,
    _contrastive_scores_fast,
    _phase,
    _structured_scores_streaming,
)
from ecup_matching.submission.v6_parallel import parallel_supported, resolve_worker_count
from ecup_matching.submission.v6_text_cache import build_dual_text_cache


CANDIDATE = "no_teacher"
GRAPH_CONFIG = {
    "reciprocal_best_bonus": 0.0,
    "reciprocal_top3_bonus": 0.0,
    "endpoint_rank_weight": 0.02,
    "ambiguity_penalty": 0.01,
}


def assert_no_teacher_assets(root: Path) -> None:
    root = Path(root)
    offenders = [
        path
        for path in root.iterdir()
        if "teacher" in path.name.lower() and path.name not in {"NO_TEACHER_KEEPER.json"}
    ] if root.exists() else []
    if offenders:
        raise RuntimeError(
            "v10 faststack must not contain teacher assets: "
            + ", ".join(sorted(path.name for path in offenders))
        )


def compose_no_teacher_signals(non_teacher: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reproduce the frozen v6 no-teacher candidate exactly.

    The missing pair-teacher signal is replaced by the unweighted mean of
    target-free percentile ranks of the five retained signals.  This operation
    uses no target labels and requires no teacher checkpoint at inference.
    """
    return build_fast_candidate_scores(non_teacher, CANDIDATE)


def predict_to_csv_v10_faststack(
    *,
    items_path: Path,
    matches_path: Path,
    structured_model_path: Path,
    contrastive_model_dir: Path,
    category_model_path: Path,
    hgb_model_path: Path,
    runtime_root: Path,
    output_path: Path,
) -> pd.DataFrame:
    started = time.perf_counter()
    previous = started
    runtime_root = Path(runtime_root)
    assert_no_teacher_assets(runtime_root)

    pairs = pd.read_parquet(matches_path, columns=["id1", "id2"])
    needed_ids = pd.unique(pd.concat([pairs["id1"], pairs["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, needed_ids, include_attributes=True)
    raw_category = items.set_index("id")["category"].astype(str)
    pairs = pairs.copy()
    pairs["category"] = pairs["id1"].map(raw_category)
    right_category = pairs["id2"].map(raw_category)
    if pairs["category"].isna().any() or right_category.isna().any():
        raise RuntimeError("failed to attach pair category")
    if not np.array_equal(
        pairs["category"].astype(str).to_numpy(), right_category.astype(str).to_numpy()
    ):
        raise RuntimeError("pair endpoints disagree on category")

    structured = joblib.load(structured_model_path)
    category_model = json.loads(category_model_path.read_text(encoding="utf-8"))
    hgb_bundle = joblib.load(hgb_model_path)
    if category_model.get("candidate") != CANDIDATE or hgb_bundle.get("candidate") != CANDIDATE:
        raise RuntimeError("v10 faststack meta artifacts are not frozen no_teacher refits")

    legacy_features, legacy_features_v2, legacy_textnorm, legacy_item_text, legacy_sparse = (
        _load_legacy_modules(runtime_root)
    )
    structured_workers = resolve_worker_count(8)
    print(
        f"[v10-faststack] pairs={len(pairs):,} items={len(items):,} "
        f"structured_workers={structured_workers} parallel={parallel_supported()}",
        flush=True,
    )
    previous = _phase("load", started, previous)

    structured_scores = _structured_scores_streaming(
        items=items,
        pairs=pairs,
        structured=structured,
        legacy_features=legacy_features,
        legacy_features_v2=legacy_features_v2,
        legacy_sparse=legacy_sparse,
        chunk_size=STRUCTURED_CHUNK_SIZE,
        workers=structured_workers,
    )
    del structured
    gc.collect()
    previous = _phase("structured", started, previous)

    # Reuse the byte-identical parallel legacy serializer.  It emits both old
    # views in one normalization pass; the teacher text strings are immediately
    # discarded and no teacher model/checkpoint is ever loaded or packaged.
    contrastive_text_cache, unused_teacher_text_cache = build_dual_text_cache(
        items,
        legacy_textnorm,
        legacy_item_text,
        workers=structured_workers,
    )
    del unused_teacher_text_cache
    gc.collect()
    previous = _phase("text_cache", started, previous)

    contrastive = _contrastive_scores_fast(
        items,
        pairs,
        contrastive_model_dir,
        legacy_textnorm,
        legacy_item_text,
        text_cache=contrastive_text_cache,
    )
    del contrastive_text_cache
    gc.collect()
    previous = _phase("contrastive", started, previous)

    non_teacher = {
        "weak": structured_scores["weak"],
        "sparse": structured_scores["sparse"],
        "explicit": structured_scores["explicit"],
        "contrastive": contrastive,
        "typed_explicit": structured_scores["typed_explicit"],
    }
    six = compose_no_teacher_signals(non_teacher)
    categories = pairs["category"].astype(str).to_numpy()
    base = category_shrunk_hgb_equal_rank_fusion(
        six,
        categories,
        category_model,
        hgb_bundle,
    )
    previous = _phase("meta", started, previous)

    gf = graph_features(pairs[["id1", "id2", "category"]], base)
    final = graph_rescore(base, gf, **GRAPH_CONFIG)
    previous = _phase("graph", started, previous)
    if len(final) != len(pairs) or not np.isfinite(final).all():
        raise RuntimeError("v10 faststack final score is incomplete or non-finite")

    # AP depends only on ranking; graph_rescore intentionally has a wider
    # numerical range than [0,1].  Do not clip, because clipping can create ties.
    result = pairs[["id1", "id2"]].copy()
    result["predict"] = final.astype(np.float64, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    _phase("write", started, previous)
    print(
        f"[v10-faststack] wrote {output_path} rows={len(result):,} "
        f"total_seconds={time.perf_counter()-started:.2f}",
        flush=True,
    )
    return result


__all__ = [
    "CANDIDATE",
    "GRAPH_CONFIG",
    "assert_no_teacher_assets",
    "compose_no_teacher_signals",
    "predict_to_csv_v10_faststack",
]
