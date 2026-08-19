from __future__ import annotations

import gc
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import time
import traceback

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
from ecup_matching.submission.v10_text_cache import build_contrastive_text_cache
from ecup_matching.submission.v6_parallel import parallel_supported, resolve_worker_count


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
    """Reproduce the frozen v6 no-teacher candidate exactly."""
    return build_fast_candidate_scores(non_teacher, CANDIDATE)


def _structured_process_entry(
    result_queue,
    *,
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    structured: dict,
    legacy_features,
    legacy_features_v2,
    legacy_sparse,
    workers: int,
) -> None:
    """Run the CPU-heavy structured branch outside the CUDA-owning process."""
    started = time.perf_counter()
    try:
        scores = _structured_scores_streaming(
            items=items,
            pairs=pairs,
            structured=structured,
            legacy_features=legacy_features,
            legacy_features_v2=legacy_features_v2,
            legacy_sparse=legacy_sparse,
            chunk_size=STRUCTURED_CHUNK_SIZE,
            workers=workers,
        )
        result_queue.put(("ok", scores, float(time.perf_counter() - started)))
    except BaseException:
        result_queue.put(("error", traceback.format_exc(), float(time.perf_counter() - started)))


def _receive_structured(proc, result_queue) -> tuple[dict[str, np.ndarray], float]:
    while True:
        try:
            status, payload, seconds = result_queue.get(timeout=1.0)
            break
        except Empty:
            if not proc.is_alive():
                proc.join(timeout=1.0)
                raise RuntimeError(
                    f"v10 structured worker exited with code {proc.exitcode} without a result"
                )
    proc.join(timeout=10.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
        raise RuntimeError("v10 structured worker did not exit after returning its result")
    if status != "ok":
        raise RuntimeError(f"v10 structured worker failed:\n{payload}")
    if proc.exitcode not in (0, None):
        raise RuntimeError(f"v10 structured worker exit code {proc.exitcode}")
    return payload, float(seconds)


def _text_worker_count(structured_workers: int) -> int:
    override = os.environ.get("ECUP_V10_TEXT_WORKERS", "").strip()
    if override:
        try:
            return max(1, min(int(override), 4))
        except ValueError:
            pass
    # Text serialization overlaps the structured worker pool only briefly. Three
    # workers keeps that overlap bounded while bringing the GPU online quickly.
    return max(1, min(3, int(structured_workers)))


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
    structured_workers = resolve_worker_count()
    text_workers = _text_worker_count(structured_workers)
    use_overlap = parallel_supported() and "fork" in multiprocessing.get_all_start_methods()
    print(
        f"[v10-faststack] pairs={len(pairs):,} items={len(items):,} "
        f"structured_workers={structured_workers} text_workers={text_workers} "
        f"cpu_gpu_overlap={use_overlap}",
        flush=True,
    )
    previous = _phase("load", started, previous)

    structured_proc = None
    structured_queue = None
    try:
        if use_overlap:
            ctx = multiprocessing.get_context("fork")
            structured_queue = ctx.Queue(maxsize=1)
            structured_proc = ctx.Process(
                target=_structured_process_entry,
                kwargs={
                    "result_queue": structured_queue,
                    "items": items,
                    "pairs": pairs,
                    "structured": structured,
                    "legacy_features": legacy_features,
                    "legacy_features_v2": legacy_features_v2,
                    "legacy_sparse": legacy_sparse,
                    "workers": structured_workers,
                },
                name="ecup-v10-structured",
            )
            structured_proc.start()
            print(
                f"[v10-faststack] structured child pid={structured_proc.pid} started before CUDA init",
                flush=True,
            )
        else:
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

        cache_started = time.perf_counter()
        contrastive_text_cache = build_contrastive_text_cache(
            items,
            legacy_textnorm,
            legacy_item_text,
            workers=text_workers,
        )
        cache_seconds = time.perf_counter() - cache_started
        print(
            f"[v10-faststack] contrastive_text_cache seconds={cache_seconds:.3f}",
            flush=True,
        )
        previous = _phase("text_cache", started, previous)

        contrastive_started = time.perf_counter()
        contrastive = _contrastive_scores_fast(
            items,
            pairs,
            contrastive_model_dir,
            legacy_textnorm,
            legacy_item_text,
            text_cache=contrastive_text_cache,
        )
        contrastive_seconds = time.perf_counter() - contrastive_started
        del contrastive_text_cache
        gc.collect()
        previous = _phase("contrastive", started, previous)

        if structured_proc is not None:
            structured_scores, structured_seconds = _receive_structured(
                structured_proc, structured_queue
            )
            structured_proc = None
            print(
                f"[v10-faststack] overlap_complete structured_seconds={structured_seconds:.3f} "
                f"text_plus_contrastive_seconds={cache_seconds + contrastive_seconds:.3f} "
                f"critical_seconds={max(structured_seconds, cache_seconds + contrastive_seconds):.3f}",
                flush=True,
            )
        else:
            structured_seconds = float("nan")
    except BaseException:
        if structured_proc is not None and structured_proc.is_alive():
            structured_proc.terminate()
            structured_proc.join(timeout=5.0)
        raise
    finally:
        if structured_queue is not None:
            structured_queue.close()
    del structured
    gc.collect()
    previous = _phase("parallel_features_ready", started, previous)

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
