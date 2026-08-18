"""Measure and gate a low-LR post-human weak refresh on one checkpoint path."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .run_v5_pretrained_biencoder import development_rows_and_folds
from .run_v7_outer_oof import IMMUTABLE_SPLIT_SHA, _phase, _stream_text_cache
from .run_v7_outer_oof_fast import _load_model_no_checkpoint
from .run_v7_outer_oof_frozen import _load_immutable_manifest
from .train_v4_reranker import DEFAULT_MODEL_REVISION, _verify_model_revision
from .v5_evaluation import macro_ap_report
from .v7_runtime import predict_pairs
from .v17_weak_holdout import split_weak_item_disjoint
from .v18_neural import train_pair_phase_v18
from .run_v18_probe import (
    _active_learning_export,
    _prepare_candidate_weak,
    _score_weak,
)
from .v19_refresh_gate import evaluate_refresh


def _human_score(model, tokenizer, frame, texts, *, max_length: int) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    score, inference = predict_pairs(
        model=model,
        tokenizer=tokenizer,
        frame=frame,
        texts=texts,
        device="cuda",
        max_length=max_length,
        batch_size=64,
    )
    report = macro_ap_report(frame, score)
    return np.asarray(score, dtype=np.float64), report, inference


def run_v19_refresh_probe(
    *,
    fold: int,
    human_items_path: Path,
    human_matches_path: Path,
    weak_matches_path: Path,
    full_items_path: Path,
    output_dir: Path,
    model_path: str,
    base_model_revision: str,
    expected_split_sha: str = IMMUTABLE_SPLIT_SHA,
    max_length: int = 256,
    max_chars: int = 900,
    weak_presample_rows: int = 1_200_000,
    weak_final_rows: int = 600_000,
    weak_epochs: float = 0.35,
    weak_holdout_fraction: float = 0.05,
    human_epochs: float = 1.0,
    refresh_epochs: float = 0.05,
    learning_rate: float = 1.5e-5,
    weak_learning_rate: float = 1.0e-5,
    refresh_learning_rate: float = 2.0e-6,
    ranking_weight: float = 0.25,
    seed: int = 2026,
) -> dict[str, object]:
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("canonical v19 probe requires CUDA")
    if expected_split_sha != IMMUTABLE_SPLIT_SHA:
        raise ValueError("v19 probe may only use the immutable split SHA")
    if int(fold) not in range(5):
        raise ValueError("fold must be one of 0..4")
    if abs(float(refresh_epochs) - 0.05) > 1e-12:
        raise ValueError("v19 refresh_epochs is preregistered at 0.05")
    if abs(float(refresh_learning_rate) - 2.0e-6) > 1e-15:
        raise ValueError("v19 refresh_learning_rate is preregistered at 2e-6")
    _verify_model_revision(model_path, base_model_revision)
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    human_items = pd.read_parquet(human_items_path, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    pairs, manifest, overlap = _load_immutable_manifest(
        human_items, matches, expected_split_sha=expected_split_sha
    )
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(matches))
    if len(dev_rows) != 285_210 or len(manifest.get("gold_rows", [])) != 80_444:
        raise RuntimeError("immutable split row counts changed")
    dev = pairs.iloc[dev_rows].reset_index(drop=True)
    train_mask = fold_ids != int(fold)
    held_mask = fold_ids == int(fold)
    human_train = dev.loc[train_mask, ["id1", "id2", "target", "category"]].reset_index(drop=True)
    held = dev.loc[held_mask, ["id1", "id2", "target", "category"]].reset_index(drop=True)
    held_rows = dev_rows[held_mask]
    category_row_counts = {
        str(key): int(value)
        for key, value in held["category"].astype(str).value_counts().to_dict().items()
    }

    human_item_universe = set(matches["id1"].tolist()) | set(matches["id2"].tolist())
    weak, weak_texts, weak_report = _prepare_candidate_weak(
        weak_matches_path=weak_matches_path,
        full_items_path=full_items_path,
        forbidden_human_item_ids=human_item_universe,
        weak_presample_rows=weak_presample_rows,
        weak_final_rows=weak_final_rows,
        max_chars=max_chars,
        seed=seed,
        quality=False,
    )
    weak = weak.copy()
    weak["soft_target"] = pd.to_numeric(weak["target"], errors="raise").astype(float)
    weak_train, weak_held, holdout_report = split_weak_item_disjoint(
        weak, holdout_fraction=weak_holdout_fraction, seed=seed + 977
    )
    del weak
    gc.collect()
    if int(holdout_report["item_overlap"]) != 0:
        raise RuntimeError("weak holdout item overlap")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = _load_model_no_checkpoint(model_path, last_n_layers=8, device="cuda")
    _phase(
        "v19-refresh-start",
        fold=int(fold),
        weak_final_rows=int(weak_final_rows),
        weak_epochs=float(weak_epochs),
        refresh_epochs=float(refresh_epochs),
        refresh_learning_rate=float(refresh_learning_rate),
        cuda_device=torch.cuda.get_device_name(0),
    )

    weak_training = train_pair_phase_v18(
        model=model,
        tokenizer=tokenizer,
        frame=weak_train,
        texts=weak_texts,
        device="cuda",
        phase="v19-weak",
        epochs=weak_epochs,
        physical_batch_size=32,
        effective_batch_size=32,
        max_length=max_length,
        learning_rate=weak_learning_rate,
        ranking_weight=ranking_weight,
        seed=seed,
        weak=True,
        telemetry_every_steps=100,
    )
    _, weak_after_weak = _score_weak(
        model,
        tokenizer,
        weak_held,
        weak_texts,
        device="cuda",
        max_length=max_length,
        stage="v19-after-weak",
    )

    needed = (
        set(human_train["id1"].tolist()) | set(human_train["id2"].tolist())
        | set(held["id1"].tolist()) | set(held["id2"].tolist())
    )
    fold_items = human_items[human_items["id"].isin(needed)].copy().reset_index(drop=True)
    if set(fold_items["id"].tolist()) != needed:
        raise RuntimeError("v19 human item subset is incomplete")
    human_texts = _stream_text_cache(fold_items, max_chars=max_chars)
    human_training = train_pair_phase_v18(
        model=model,
        tokenizer=tokenizer,
        frame=human_train,
        texts=human_texts,
        device="cuda",
        phase=f"v19-human-fold{int(fold)}",
        epochs=human_epochs,
        physical_batch_size=32,
        effective_batch_size=32,
        max_length=max_length,
        learning_rate=learning_rate,
        ranking_weight=ranking_weight,
        seed=seed + 100,
        weak=False,
        telemetry_every_steps=100,
    )

    pre_human_score, pre_human_report, pre_human_inference = _human_score(
        model, tokenizer, held, human_texts, max_length=max_length
    )
    _, pre_weak_report = _score_weak(
        model,
        tokenizer,
        weak_held,
        weak_texts,
        device="cuda",
        max_length=max_length,
        stage="v19-pre-refresh",
    )
    pre_refresh = {
        "human_macro_average_precision": float(pre_human_report["macro_average_precision"]),
        "weak_macro_average_precision": float(pre_weak_report["macro_average_precision"]),
        "weak_soft_brier": float(pre_weak_report["soft_brier"]),
        "weak_soft_cross_entropy": float(pre_weak_report["soft_cross_entropy"]),
        "per_category_ap": pre_human_report["per_category_ap"],
        "category_row_counts": category_row_counts,
        "gold_metric_opened": False,
        "cross_split_item_overlap": int(overlap["cross_split_item_overlap"]),
    }

    refresh_training = train_pair_phase_v18(
        model=model,
        tokenizer=tokenizer,
        frame=weak_train,
        texts=weak_texts,
        device="cuda",
        phase="v19-post-human-weak-refresh",
        epochs=refresh_epochs,
        physical_batch_size=32,
        effective_batch_size=32,
        max_length=max_length,
        learning_rate=refresh_learning_rate,
        ranking_weight=ranking_weight,
        seed=seed + 200,
        weak=True,
        telemetry_every_steps=100,
    )

    post_human_score, post_human_report, post_human_inference = _human_score(
        model, tokenizer, held, human_texts, max_length=max_length
    )
    post_weak_score, post_weak_report = _score_weak(
        model,
        tokenizer,
        weak_held,
        weak_texts,
        device="cuda",
        max_length=max_length,
        stage="v19-post-refresh",
    )
    post_refresh = {
        "human_macro_average_precision": float(post_human_report["macro_average_precision"]),
        "weak_macro_average_precision": float(post_weak_report["macro_average_precision"]),
        "weak_soft_brier": float(post_weak_report["soft_brier"]),
        "weak_soft_cross_entropy": float(post_weak_report["soft_cross_entropy"]),
        "per_category_ap": post_human_report["per_category_ap"],
        "category_row_counts": category_row_counts,
        "gold_metric_opened": False,
        "cross_split_item_overlap": int(overlap["cross_split_item_overlap"]),
    }
    refresh_gate = evaluate_refresh(pre_refresh, post_refresh)
    active_learning = _active_learning_export(
        weak_held, post_weak_score, output_dir / "active-learning.csv"
    )

    payload: dict[str, object] = {
        "version": "v19-antiforget-refresh-probe-v1",
        "diagnostic_only": True,
        "fold": int(fold),
        "base_model": "ai-forever/ruBert-base",
        "base_model_revision": base_model_revision.lower(),
        "cuda_device": torch.cuda.get_device_name(0),
        "max_length": int(max_length),
        "max_chars": int(max_chars),
        "weak_presample_rows": int(weak_presample_rows),
        "weak_final_rows": int(weak_final_rows),
        "weak_epochs": float(weak_epochs),
        "weak_holdout_fraction": float(weak_holdout_fraction),
        "human_epochs": float(human_epochs),
        "refresh_epochs": float(refresh_epochs),
        "refresh_learning_rate": float(refresh_learning_rate),
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev_rows)),
        "sealed_gold_rows": int(len(manifest["gold_rows"])),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "cross_split_item_overlap": int(overlap["cross_split_item_overlap"]),
        "train_rows": int(len(human_train)),
        "held_rows": int(len(held)),
        "weak_preparation": weak_report,
        "weak_holdout": holdout_report,
        "weak_after_weak_phase": weak_after_weak,
        "pre_refresh": pre_refresh,
        "post_refresh": post_refresh,
        "refresh_gate": refresh_gate,
        "weak_training": weak_training.__dict__,
        "human_training": human_training.__dict__,
        "refresh_training": refresh_training.__dict__,
        "pre_human_inference": pre_human_inference,
        "post_human_inference": post_human_inference,
        "active_learning": active_learning,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    pd.DataFrame(
        {
            "row_index": held_rows,
            "fold": int(fold),
            "pre_refresh_score": pre_human_score,
            "post_refresh_score": post_human_score,
        }
    ).to_parquet(output_dir / "v19-fold-oof.parquet", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("V19_REFRESH_PROBE=" + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--human-items", type=Path, required=True)
    parser.add_argument("--human-matches", type=Path, required=True)
    parser.add_argument("--llm-matches", type=Path, required=True)
    parser.add_argument("--full-items", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="/opt/models/rubert-base")
    parser.add_argument("--base-model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--split-sha", default=IMMUTABLE_SPLIT_SHA)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--weak-presample-rows", type=int, default=1_200_000)
    parser.add_argument("--weak-final-rows", type=int, default=600_000)
    parser.add_argument("--weak-epochs", type=float, default=0.35)
    parser.add_argument("--weak-holdout-fraction", type=float, default=0.05)
    parser.add_argument("--human-epochs", type=float, default=1.0)
    parser.add_argument("--refresh-epochs", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--weak-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--refresh-learning-rate", type=float, default=2.0e-6)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run_v19_refresh_probe(
        fold=args.fold,
        human_items_path=args.human_items,
        human_matches_path=args.human_matches,
        weak_matches_path=args.llm_matches,
        full_items_path=args.full_items,
        output_dir=args.output_dir,
        model_path=args.base_model,
        base_model_revision=args.base_model_revision,
        expected_split_sha=args.split_sha,
        max_length=args.max_length,
        max_chars=args.max_chars,
        weak_presample_rows=args.weak_presample_rows,
        weak_final_rows=args.weak_final_rows,
        weak_epochs=args.weak_epochs,
        weak_holdout_fraction=args.weak_holdout_fraction,
        human_epochs=args.human_epochs,
        refresh_epochs=args.refresh_epochs,
        learning_rate=args.learning_rate,
        weak_learning_rate=args.weak_learning_rate,
        refresh_learning_rate=args.refresh_learning_rate,
        ranking_weight=args.ranking_weight,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
