"""Full-development v18 production refit for a pre-promoted mechanism set."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from .run_v5_pretrained_biencoder import development_rows_and_folds
from .run_v7_outer_oof import IMMUTABLE_SPLIT_SHA, _phase, _stream_text_cache
from .run_v7_outer_oof_fast import _load_model_no_checkpoint
from .run_v7_outer_oof_frozen import _load_immutable_manifest
from .train_v4_reranker import DEFAULT_MODEL_REVISION, _verify_model_revision
from .v7_runtime import predict_pairs
from .v18_ema import ExponentialMovingAverage
from .v18_hard_mining import select_disagreement_hard_examples
from .v18_neural import train_pair_phase_v18
from .run_v18_probe import (
    _candidate_mechanisms,
    _epoch_capacity,
    _epochs_for_examples,
    _prepare_candidate_weak,
)
from .v18_weak_quality import split_weak_curriculum
from .weak_labels import sample_weak_training


def run_v18_production(
    *,
    mechanisms_json: str,
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
    weak_presample_rows: int = 3_000_000,
    weak_final_rows: int = 1_500_000,
    weak_epochs: float = 1.0,
    human_epochs: float = 1.0,
    physical_batch_size: int = 32,
    effective_batch_size: int = 32,
    learning_rate: float = 1.5e-5,
    weak_learning_rate: float = 1.0e-5,
    ranking_weight: float = 0.25,
    seed: int = 2026,
) -> dict[str, object]:
    import torch
    from transformers import AutoTokenizer

    mechanisms = _candidate_mechanisms("combined", mechanisms_json)
    if not torch.cuda.is_available():
        raise RuntimeError("v18 production refit requires CUDA")
    if expected_split_sha != IMMUTABLE_SPLIT_SHA:
        raise ValueError("v18 production may only use the immutable split SHA")
    _verify_model_revision(model_path, base_model_revision)
    started = time.perf_counter()
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_layers = 12 if "full" in mechanisms else 8
    phase_batch = 16 if "full" in mechanisms else int(physical_batch_size)
    phase_effective = max(int(effective_batch_size), 32) if "full" in mechanisms else int(effective_batch_size)
    human_lr = 8e-6 if "full" in mechanisms else float(learning_rate)
    weak_lr = 8e-6 if "full" in mechanisms else float(weak_learning_rate)
    view_kwargs = {
        "pair_swap_probability": 0.5 if "views" in mechanisms else 0.0,
        "residual_dropout_probability": 0.15 if "views" in mechanisms else 0.0,
        "numeric_dropout_probability": 0.05 if "views" in mechanisms else 0.0,
    }
    _phase(
        "v18-production-start",
        mechanisms=list(mechanisms),
        weak_final_rows=int(weak_final_rows),
        weak_epochs=float(weak_epochs),
        last_layers=int(last_layers),
        cuda_device=torch.cuda.get_device_name(0),
    )

    human_items = pd.read_parquet(human_items_path, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    pairs, manifest, overlap = _load_immutable_manifest(
        human_items, matches, expected_split_sha=expected_split_sha
    )
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(matches))
    if len(dev_rows) != 285_210 or len(fold_ids) != 285_210 or len(manifest.get("gold_rows", [])) != 80_444:
        raise RuntimeError("immutable split row counts changed")
    human_train = pairs.iloc[dev_rows][["id1", "id2", "target", "category"]].reset_index(drop=True)
    human_item_universe = set(matches["id1"].tolist()) | set(matches["id2"].tolist())

    weak, weak_texts, weak_report = _prepare_candidate_weak(
        weak_matches_path=weak_matches_path,
        full_items_path=full_items_path,
        forbidden_human_item_ids=human_item_universe,
        weak_presample_rows=weak_presample_rows,
        weak_final_rows=weak_final_rows,
        max_chars=max_chars,
        seed=seed,
        quality="quality" in mechanisms,
    )
    if (set(weak["id1"].tolist()) | set(weak["id2"].tolist())) & human_item_universe:
        raise RuntimeError("production weak data leaked human items")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = _load_model_no_checkpoint(model_path, last_n_layers=last_layers, device="cuda")
    ema = ExponentialMovingAverage(model, decay=0.999) if "ema" in mechanisms else None
    desired_weak_examples = max(
        1,
        int(round(_epoch_capacity(weak, phase_batch, seed) * float(weak_epochs))),
    )
    weak_phase_results: list[dict[str, object]] = []
    hard_report: dict[str, object] | None = None
    curriculum_report: dict[str, object] | None = None

    if "quality" in mechanisms or "hard" in mechanisms:
        if "quality" in mechanisms:
            high, broad, curriculum_report = split_weak_curriculum(weak, high_margin=0.30)
            first_frame = high
        else:
            broad = weak
            first_frame = weak
            curriculum_report = {"high_rows": int(len(weak)), "broad_rows": int(len(weak))}
        first_fraction = 0.40 if "hard" in mechanisms else 0.35
        first_examples = max(1, int(round(desired_weak_examples * first_fraction)))
        first_epochs = _epochs_for_examples(
            first_frame, batch_size=phase_batch, seed=seed, examples=first_examples
        )
        first = train_pair_phase_v18(
            model=model,
            tokenizer=tokenizer,
            frame=first_frame,
            texts=weak_texts,
            device="cuda",
            phase="v18-production-weak-phase1",
            epochs=first_epochs,
            physical_batch_size=phase_batch,
            effective_batch_size=phase_effective,
            max_length=max_length,
            learning_rate=weak_lr,
            ranking_weight=ranking_weight,
            seed=seed,
            weak=True,
            ema=ema,
            **view_kwargs,
        )
        weak_phase_results.append(first.__dict__)
        remaining_examples = max(1, desired_weak_examples - int(first.examples_seen))
        if "hard" in mechanisms:
            mining_pool = sample_weak_training(
                broad, max_rows=min(200_000, len(broad)), seed=seed + 1801
            )
            mining_score, mining_inference = predict_pairs(
                model=model,
                tokenizer=tokenizer,
                frame=mining_pool,
                texts=weak_texts,
                device="cuda",
                max_length=max_length,
                batch_size=64,
            )
            hard_rows = min(120_000, max(20_000, len(mining_pool) // 2))
            second_frame, hard_report = select_disagreement_hard_examples(
                mining_pool, mining_score, max_rows=hard_rows, seed=seed + 1802
            )
            hard_report["mining_inference"] = mining_inference
            del mining_pool, mining_score
        else:
            second_frame = broad
        second_epochs = _epochs_for_examples(
            second_frame, batch_size=phase_batch, seed=seed + 1, examples=remaining_examples
        )
        second = train_pair_phase_v18(
            model=model,
            tokenizer=tokenizer,
            frame=second_frame,
            texts=weak_texts,
            device="cuda",
            phase="v18-production-weak-phase2",
            epochs=second_epochs,
            physical_batch_size=phase_batch,
            effective_batch_size=phase_effective,
            max_length=max_length,
            learning_rate=weak_lr,
            ranking_weight=ranking_weight,
            seed=seed + 1,
            weak=True,
            ema=ema,
            **view_kwargs,
        )
        weak_phase_results.append(second.__dict__)
    else:
        one = train_pair_phase_v18(
            model=model,
            tokenizer=tokenizer,
            frame=weak,
            texts=weak_texts,
            device="cuda",
            phase="v18-production-weak",
            epochs=weak_epochs,
            physical_batch_size=phase_batch,
            effective_batch_size=phase_effective,
            max_length=max_length,
            learning_rate=weak_lr,
            ranking_weight=ranking_weight,
            seed=seed,
            weak=True,
            ema=ema,
            **view_kwargs,
        )
        weak_phase_results.append(one.__dict__)
    del weak, weak_texts
    gc.collect()
    torch.cuda.empty_cache()

    needed = set(human_train["id1"].tolist()) | set(human_train["id2"].tolist())
    dev_items = human_items[human_items["id"].isin(needed)].copy().reset_index(drop=True)
    if set(dev_items["id"].tolist()) != needed:
        raise RuntimeError("production human item subset is incomplete")
    human_texts = _stream_text_cache(dev_items, max_chars=max_chars)
    human_training = train_pair_phase_v18(
        model=model,
        tokenizer=tokenizer,
        frame=human_train,
        texts=human_texts,
        device="cuda",
        phase="v18-production-human-all-folds",
        epochs=human_epochs,
        physical_batch_size=phase_batch,
        effective_batch_size=phase_effective,
        max_length=max_length,
        learning_rate=human_lr,
        ranking_weight=ranking_weight,
        seed=seed + 100,
        weak=False,
        ema=ema,
        **view_kwargs,
    )
    if ema is not None:
        ema.copy_to(model)

    model_dir = output_dir / "model_v7_teacher"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    saved = sorted(path.name for path in model_dir.iterdir() if path.is_file())
    weights = [name for name in saved if name.endswith(".safetensors")]
    if len(weights) != 1:
        raise RuntimeError(f"expected exactly one production safetensors checkpoint, got {weights}")

    payload: dict[str, object] = {
        "version": "v18-production-refit",
        "candidate": "v18-strengthened-rubert-crossencoder",
        "mechanisms": list(mechanisms),
        "is_production_refit": True,
        "validation_metric_reported": False,
        "base_model": "ai-forever/ruBert-base",
        "base_model_revision": base_model_revision.lower(),
        "cuda_device": torch.cuda.get_device_name(0),
        "max_length": int(max_length),
        "max_chars": int(max_chars),
        "inference_batch_size": 64,
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev_rows)),
        "training_rows": int(len(human_train)),
        "sealed_gold_rows": int(len(manifest["gold_rows"])),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "cross_split_item_overlap": int(overlap["cross_split_item_overlap"]),
        "weak_presample_rows": int(weak_presample_rows),
        "weak_final_rows": int(weak_final_rows),
        "weak_epochs": float(weak_epochs),
        "weak_examples_budget": int(desired_weak_examples),
        "weak_examples_seen": int(sum(int(value["examples_seen"]) for value in weak_phase_results)),
        "human_epochs": float(human_epochs),
        "physical_batch_size": int(phase_batch),
        "effective_batch_size": int(phase_effective),
        "trainable_last_layers": int(last_layers),
        "learning_rate": float(human_lr),
        "weak_learning_rate": float(weak_lr),
        "weak_preparation": weak_report,
        "weak_curriculum": curriculum_report,
        "hard_mining": hard_report,
        "weak_training_phases": weak_phase_results,
        "human_training": human_training.__dict__,
        "ema_decay": 0.999 if ema is not None else None,
        "saved_files": saved,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "production-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("V18_PRODUCTION=" + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanisms-json", required=True)
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
    parser.add_argument("--weak-presample-rows", type=int, default=3_000_000)
    parser.add_argument("--weak-final-rows", type=int, default=1_500_000)
    parser.add_argument("--weak-epochs", type=float, default=1.0)
    parser.add_argument("--human-epochs", type=float, default=1.0)
    parser.add_argument("--physical-batch-size", type=int, default=32)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--weak-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run_v18_production(
        mechanisms_json=args.mechanisms_json,
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
        human_epochs=args.human_epochs,
        physical_batch_size=args.physical_batch_size,
        effective_batch_size=args.effective_batch_size,
        learning_rate=args.learning_rate,
        weak_learning_rate=args.weak_learning_rate,
        ranking_weight=args.ranking_weight,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
