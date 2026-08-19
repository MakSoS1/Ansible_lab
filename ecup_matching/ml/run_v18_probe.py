"""Generic v18 probe for orthogonal training/data ablations.

The retained inference family stays unchanged. This module only changes how the
single RuBERT pair CrossEncoder is trained and how candidates are validated.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .run_v7_outer_oof import IMMUTABLE_SPLIT_SHA, _phase, _stream_text_cache
from .run_v7_outer_oof_fast import _load_model_no_checkpoint
from .run_v7_outer_oof_frozen import _load_immutable_manifest
from .train_v1 import attach_pair_category
from .train_v2_structured import prefilter_weak_candidates_parquet
from .train_v4_reranker import DEFAULT_MODEL_REVISION, _verify_model_revision
from .v5_evaluation import macro_ap_report
from .v7_neural import MacroPairBatchSampler
from .v7_runtime import build_v7_text_cache_from_parquet, predict_pairs
from .v17_weak_holdout import split_weak_item_disjoint
from .v18_ema import ExponentialMovingAverage
from .v18_hard_mining import select_disagreement_hard_examples
from .v18_neural import train_pair_phase_v18
from .v18_weak_quality import prepare_weak_pairs_v18, split_weak_curriculum
from .weak_labels import prepare_weak_pairs, sample_weak_training


_SINGLE_MECHANISMS = {
    "control": (),
    "q1-quality": ("quality",),
    "q2-hard": ("hard",),
    "q3-views": ("views",),
    "q4-full": ("full",),
    "q5-ema": ("ema",),
}
_ALLOWED = {"quality", "hard", "views", "full", "ema"}


def _candidate_mechanisms(candidate: str, mechanisms_json: str | None) -> tuple[str, ...]:
    if candidate in _SINGLE_MECHANISMS:
        if mechanisms_json:
            raise ValueError("mechanisms-json is only valid for combined")
        return tuple(_SINGLE_MECHANISMS[candidate])
    if candidate != "combined":
        raise ValueError(f"unsupported v18 candidate {candidate!r}")
    raw = json.loads(mechanisms_json or "[]")
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise ValueError("combined mechanisms-json must be a JSON string list")
    values = tuple(sorted(set(raw)))
    unknown = set(values) - _ALLOWED
    if unknown:
        raise ValueError(f"unknown mechanisms: {sorted(unknown)}")
    if not values:
        raise ValueError("combined candidate requires at least one mechanism")
    return values


def _prepare_candidate_weak(
    *,
    weak_matches_path: Path,
    full_items_path: Path,
    forbidden_human_item_ids: set[object],
    weak_presample_rows: int,
    weak_final_rows: int,
    max_chars: int,
    seed: int,
    quality: bool,
) -> tuple[pd.DataFrame, dict[object, str], dict[str, object]]:
    _phase(
        "v18-weak-prefilter",
        quality=bool(quality),
        forbidden_human_items=len(forbidden_human_item_ids),
        max_presample_rows=int(weak_presample_rows),
    )
    weak, input_rows = prefilter_weak_candidates_parquet(
        weak_matches_path,
        validation_item_ids=forbidden_human_item_ids,
        max_presample_rows=weak_presample_rows,
        seed=seed,
    )
    raw = weak[["id1", "id2", "target"]]
    if quality:
        weak, prep = prepare_weak_pairs_v18(raw)
    else:
        weak, prep = prepare_weak_pairs(raw)
    endpoint_overlap = (set(weak["id1"].tolist()) | set(weak["id2"].tolist())) & forbidden_human_item_ids
    if endpoint_overlap:
        raise RuntimeError(f"v18 weak corpus contains {len(endpoint_overlap)} human items")

    intermediate_cap = min(len(weak), max(weak_final_rows, int(round(weak_final_rows * 1.5))))
    weak = sample_weak_training(weak, max_rows=intermediate_cap, seed=seed)
    weak_ids = set(weak["id1"].tolist()) | set(weak["id2"].tolist())
    category_items = select_items_by_ids(full_items_path, weak_ids, include_attributes=False)
    weak = attach_pair_category(weak, category_items)
    weak = sample_weak_training(weak, max_rows=weak_final_rows, seed=seed + 17)
    final_ids = set(weak["id1"].tolist()) | set(weak["id2"].tolist())
    if final_ids & forbidden_human_item_ids:
        raise RuntimeError("forbidden human item survived v18 weak selection")

    weak_texts, category_by_id = build_v7_text_cache_from_parquet(
        full_items_path, final_ids, max_chars=max_chars
    )
    left_category = weak["id1"].map(category_by_id)
    right_category = weak["id2"].map(category_by_id)
    if left_category.isna().any() or right_category.isna().any():
        raise RuntimeError("full-attribute weak scan missed selected IDs")
    if (left_category.astype(str) != right_category.astype(str)).any():
        raise RuntimeError("v18 weak pair crossed category")
    if (weak["category"].astype(str) != left_category.astype(str)).any():
        raise RuntimeError("v18 weak category changed between scans")
    del category_items, category_by_id
    report: dict[str, object] = {
        "input_rows": int(input_rows),
        "prepared": prep,
        "selected_rows": int(len(weak)),
        "selected_items": int(len(weak_texts)),
        "quality_mode": bool(quality),
        "human_item_overlap": 0,
    }
    keep = ["id1", "id2", "target", "category", "weak_weight"]
    for optional in ("hard_target", "_weak_margin"):
        if optional in weak.columns:
            keep.append(optional)
    return weak[keep].reset_index(drop=True), weak_texts, report


def _epoch_capacity(frame: pd.DataFrame, batch_size: int, seed: int) -> int:
    return int(len(MacroPairBatchSampler(frame, batch_size, seed)) * int(batch_size))


def _epochs_for_examples(frame: pd.DataFrame, *, batch_size: int, seed: int, examples: int) -> float:
    capacity = _epoch_capacity(frame, batch_size, seed)
    return max(1e-6, float(examples) / max(capacity, 1))


def _score_weak(model, tokenizer, frame, texts, *, device: str, max_length: int, stage: str):
    score, inference = predict_pairs(
        model=model,
        tokenizer=tokenizer,
        frame=frame,
        texts=texts,
        device=device,
        max_length=max_length,
        batch_size=64,
    )
    report = macro_ap_report(frame, score)
    payload: dict[str, object] = {
        "stage": stage,
        "rows": int(len(frame)),
        "macro_average_precision": float(report["macro_average_precision"]),
        "per_category_ap": report["per_category_ap"],
        "inference": inference,
    }
    if "soft_target" in frame.columns:
        soft = np.clip(pd.to_numeric(frame["soft_target"], errors="raise").to_numpy(float), 0.0, 1.0)
        prob = np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
        payload["soft_brier"] = float(np.mean((prob - soft) ** 2))
        payload["soft_cross_entropy"] = float(
            -np.mean(soft * np.log(prob) + (1.0 - soft) * np.log(1.0 - prob))
        )
    _phase(
        f"v18-weak-{stage}",
        **{key: value for key, value in payload.items() if key not in {"per_category_ap", "inference"}},
    )
    return np.asarray(score, dtype=np.float64), payload


def _capture_trainable(model) -> dict[str, object]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable(model, state: dict[str, object]) -> None:
    import torch

    with torch.no_grad():
        seen: set[str] = set()
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
                seen.add(name)
        if seen != set(state):
            raise RuntimeError("trainable state/model mismatch")


def _active_learning_export(frame: pd.DataFrame, score: np.ndarray, output: Path, *, max_rows: int = 10_000) -> dict[str, object]:
    work = frame.copy().reset_index(drop=True)
    work["prediction"] = np.asarray(score, dtype=float)
    reference = work["soft_target"] if "soft_target" in work.columns else work["target"]
    work["disagreement"] = (work["prediction"] - pd.to_numeric(reference, errors="raise").astype(float)).abs()
    if "hard_target" not in work.columns:
        work["hard_target"] = (pd.to_numeric(reference, errors="raise").astype(float) >= 0.5).astype(np.int8)
    work["reason"] = "candidate-vs-weak-disagreement"
    groups = list(work.groupby(["category", "hard_target"], sort=True, dropna=False))
    quota = max(1, int(max_rows) // max(1, len(groups)))
    picked: list[pd.DataFrame] = []
    used: set[int] = set()
    for _, group in groups:
        take = group.sort_values("disagreement", ascending=False, kind="mergesort").head(quota)
        picked.append(take)
        used.update(take.index.tolist())
    selected = pd.concat(picked, axis=0) if picked else work.iloc[:0]
    if len(selected) < min(max_rows, len(work)):
        remaining = work.loc[~work.index.isin(used)].sort_values(
            "disagreement", ascending=False, kind="mergesort"
        )
        selected = pd.concat([selected, remaining.head(min(max_rows, len(work)) - len(selected))], axis=0)
    columns = [
        "id1", "id2", "category", "target", "hard_target", "weak_weight",
        "prediction", "disagreement", "reason",
    ]
    if "soft_target" in selected.columns:
        columns.insert(4, "soft_target")
    selected[columns].head(max_rows).to_csv(output, index=False)
    return {
        "rows": int(min(len(selected), max_rows)),
        "groups": int(len(groups)),
        "max_rows": int(max_rows),
    }


def run_v18_probe(
    *,
    candidate: str,
    mechanisms_json: str | None,
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
    physical_batch_size: int = 32,
    effective_batch_size: int = 32,
    learning_rate: float = 1.5e-5,
    weak_learning_rate: float = 1.0e-5,
    ranking_weight: float = 0.25,
    seed: int = 2026,
) -> dict[str, object]:
    import torch
    from transformers import AutoTokenizer

    mechanisms = _candidate_mechanisms(candidate, mechanisms_json)
    if not torch.cuda.is_available():
        raise RuntimeError("canonical v18 probe requires CUDA")
    if expected_split_sha != IMMUTABLE_SPLIT_SHA:
        raise ValueError("v18 probe may only use the immutable split SHA")
    if int(fold) not in range(5):
        raise ValueError("fold must be one of 0..4")
    _verify_model_revision(model_path, base_model_revision)
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

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
        "v18-probe-start",
        candidate=candidate,
        mechanisms=list(mechanisms),
        fold=int(fold),
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
    if len(dev_rows) != 285_210 or len(manifest.get("gold_rows", [])) != 80_444:
        raise RuntimeError("immutable split row counts changed")
    dev = pairs.iloc[dev_rows].reset_index(drop=True)
    train_mask = fold_ids != int(fold)
    held_mask = fold_ids == int(fold)
    human_train = dev.loc[train_mask, ["id1", "id2", "target", "category"]].reset_index(drop=True)
    held = dev.loc[held_mask, ["id1", "id2", "target", "category"]].reset_index(drop=True)
    held_rows = dev_rows[held_mask]
    category_row_counts = {
        str(key): int(value) for key, value in held["category"].astype(str).value_counts().to_dict().items()
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
        quality="quality" in mechanisms,
    )
    weak = weak.copy()
    weak["soft_target"] = pd.to_numeric(weak["target"], errors="raise").astype(float)
    weak_train, weak_held, holdout_report = split_weak_item_disjoint(
        weak, holdout_fraction=weak_holdout_fraction, seed=seed + 977
    )
    del weak
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = _load_model_no_checkpoint(model_path, last_n_layers=last_layers, device=device)
    ema = ExponentialMovingAverage(model, decay=0.999) if "ema" in mechanisms else None
    desired_weak_examples = max(
        1,
        int(round(_epoch_capacity(weak_train, phase_batch, seed) * float(weak_epochs))),
    )
    weak_phase_results: list[dict[str, object]] = []
    hard_report: dict[str, object] | None = None

    if "quality" in mechanisms or "hard" in mechanisms:
        if "quality" in mechanisms:
            high, broad, curriculum_report = split_weak_curriculum(weak_train, high_margin=0.30)
            first_frame = high
        else:
            broad = weak_train
            first_frame = weak_train
            curriculum_report = {"high_rows": int(len(weak_train)), "broad_rows": int(len(weak_train))}
        first_fraction = 0.40 if "hard" in mechanisms else 0.35
        first_examples = max(1, int(round(desired_weak_examples * first_fraction)))
        first_epochs = _epochs_for_examples(
            first_frame, batch_size=phase_batch, seed=seed, examples=first_examples
        )
        result = train_pair_phase_v18(
            model=model,
            tokenizer=tokenizer,
            frame=first_frame,
            texts=weak_texts,
            device=device,
            phase="v18-weak-phase1",
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
        weak_phase_results.append(result.__dict__)
        remaining_examples = max(1, desired_weak_examples - int(result.examples_seen))

        if "hard" in mechanisms:
            mining_pool = sample_weak_training(
                broad,
                max_rows=min(200_000, len(broad)),
                seed=seed + 1801,
            )
            mining_score, mining_inference = predict_pairs(
                model=model,
                tokenizer=tokenizer,
                frame=mining_pool,
                texts=weak_texts,
                device=device,
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
        result = train_pair_phase_v18(
            model=model,
            tokenizer=tokenizer,
            frame=second_frame,
            texts=weak_texts,
            device=device,
            phase="v18-weak-phase2",
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
        weak_phase_results.append(result.__dict__)
        del second_frame
    else:
        curriculum_report = None
        result = train_pair_phase_v18(
            model=model,
            tokenizer=tokenizer,
            frame=weak_train,
            texts=weak_texts,
            device=device,
            phase="v18-weak",
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
        weak_phase_results.append(result.__dict__)

    if ema is not None:
        raw_after_weak = _capture_trainable(model)
        ema.copy_to(model)
        weak_score_after_weak, weak_after_weak = _score_weak(
            model, tokenizer, weak_held, weak_texts, device=device, max_length=max_length, stage="after-weak-phase-ema"
        )
        _restore_trainable(model, raw_after_weak)
        del raw_after_weak
    else:
        weak_score_after_weak, weak_after_weak = _score_weak(
            model, tokenizer, weak_held, weak_texts, device=device, max_length=max_length, stage="after-weak-phase"
        )
    del weak_score_after_weak, weak_train
    gc.collect()
    torch.cuda.empty_cache()

    needed = (
        set(human_train["id1"].tolist()) | set(human_train["id2"].tolist())
        | set(held["id1"].tolist()) | set(held["id2"].tolist())
    )
    fold_items = human_items[human_items["id"].isin(needed)].copy().reset_index(drop=True)
    if set(fold_items["id"].tolist()) != needed:
        raise RuntimeError("v18 probe item subset is incomplete")
    fold_texts = _stream_text_cache(fold_items, max_chars=max_chars)
    human_training = train_pair_phase_v18(
        model=model,
        tokenizer=tokenizer,
        frame=human_train,
        texts=fold_texts,
        device=device,
        phase=f"v18-human-fold{int(fold)}",
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

    raw_diagnostics: dict[str, object] | None = None
    if ema is not None:
        raw_human_score, raw_human_inference = predict_pairs(
            model=model, tokenizer=tokenizer, frame=held, texts=fold_texts,
            device=device, max_length=max_length, batch_size=64,
        )
        raw_human_report = macro_ap_report(held, raw_human_score)
        raw_weak_score, raw_weak_report = _score_weak(
            model, tokenizer, weak_held, weak_texts, device=device, max_length=max_length, stage="after-human-phase-raw"
        )
        raw_diagnostics = {
            "human_macro_average_precision": float(raw_human_report["macro_average_precision"]),
            "weak_macro_average_precision": float(raw_weak_report["macro_average_precision"]),
            "human_inference": raw_human_inference,
        }
        del raw_human_score, raw_weak_score
        ema.copy_to(model)

    human_score, human_inference = predict_pairs(
        model=model,
        tokenizer=tokenizer,
        frame=held,
        texts=fold_texts,
        device=device,
        max_length=max_length,
        batch_size=64,
    )
    human_report = macro_ap_report(held, human_score)
    weak_score_after_human, weak_after_human = _score_weak(
        model, tokenizer, weak_held, weak_texts, device=device, max_length=max_length,
        stage="after-human-phase-ema" if ema is not None else "after-human-phase",
    )
    active_learning = _active_learning_export(
        weak_held, weak_score_after_human, output_dir / "active-learning.csv"
    )

    weak_examples_seen = int(sum(int(value["examples_seen"]) for value in weak_phase_results))
    payload: dict[str, object] = {
        "version": "v18-orthogonal-probe-v1",
        "diagnostic_only": True,
        "candidate": candidate,
        "mechanisms": list(mechanisms),
        "fold": int(fold),
        "base_model": "ai-forever/ruBert-base",
        "base_model_revision": base_model_revision.lower(),
        "cuda_device": torch.cuda.get_device_name(0),
        "max_length": int(max_length),
        "max_chars": int(max_chars),
        "weak_presample_rows": int(weak_presample_rows),
        "weak_final_rows": int(weak_final_rows),
        "weak_epochs_requested": float(weak_epochs),
        "weak_examples_budget": int(desired_weak_examples),
        "weak_examples_seen": int(weak_examples_seen),
        "human_epochs": float(human_epochs),
        "physical_batch_size": int(phase_batch),
        "effective_batch_size": int(phase_effective),
        "trainable_last_layers": int(last_layers),
        "learning_rate": float(human_lr),
        "weak_learning_rate": float(weak_lr),
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev_rows)),
        "sealed_gold_rows": int(len(manifest["gold_rows"])),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "cross_split_item_overlap": int(overlap["cross_split_item_overlap"]),
        "train_rows": int(len(human_train)),
        "held_rows": int(len(held)),
        "fold_macro_average_precision": float(human_report["macro_average_precision"]),
        "per_category_ap": human_report["per_category_ap"],
        "category_row_counts": category_row_counts,
        "weak_holdout": holdout_report,
        "weak_holdout_after_weak_phase": weak_after_weak,
        "weak_holdout_after_human_phase": weak_after_human,
        "weak_preparation": weak_report,
        "weak_curriculum": curriculum_report,
        "hard_mining": hard_report,
        "weak_training_phases": weak_phase_results,
        "human_training": human_training.__dict__,
        "ema_decay": 0.999 if ema is not None else None,
        "ema_raw_diagnostics": raw_diagnostics,
        "human_inference": human_inference,
        "active_learning": active_learning,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    pd.DataFrame(
        {
            "row_index": held_rows,
            "fold": int(fold),
            "v18_score": np.asarray(human_score, dtype=np.float64),
        }
    ).to_parquet(output_dir / "v18-fold-oof.parquet", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("V18_PROBE=" + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=[*sorted(_SINGLE_MECHANISMS), "combined"])
    parser.add_argument("--mechanisms-json")
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
    parser.add_argument("--physical-batch-size", type=int, default=32)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--weak-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run_v18_probe(
        candidate=args.candidate,
        mechanisms_json=args.mechanisms_json,
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
