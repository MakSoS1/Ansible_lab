from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .category_attrs import learn_attribute_importance
from .data_subset import select_items_by_ids
from .label_graph import augment_transitive_positives, clean_human_pairs
from .metrics import macro_average_precision
from .reranker_data import build_reranker_examples
from .train_v1 import attach_pair_category, category_equalizing_weights
from .v2_split import fixed_v1_split
from .weak_labels import prepare_weak_pairs, remove_human_conflicts, sample_weak_training


SEED = 2026
V1_MACRO_AP = 0.49616548946964434
DEFAULT_MODEL = "cointegrated/rubert-tiny2"


def _require_torch_transformers():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover - GPU runtime only
        raise RuntimeError(
            "train_reranker_v2 requires torch and transformers; install them in the GPU runtime"
        ) from exc
    return torch, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _soft_category_weights(categories: pd.Series, source: pd.Series, weak_weight: np.ndarray) -> np.ndarray:
    cat = category_equalizing_weights(categories.reset_index(drop=True).astype(str))
    src = source.reset_index(drop=True).astype(str).to_numpy()
    base = np.where(src == "human", 10.0, np.asarray(weak_weight, dtype=float))
    weight = cat * base
    mean = float(weight.mean()) if len(weight) else 1.0
    return weight / mean if mean > 0 else weight


def _weak_prefilter_mask(
    weak: pd.DataFrame, validation_item_ids: set[object]
) -> np.ndarray:
    p = pd.to_numeric(weak["target"], errors="raise").astype(float)
    weight = np.zeros(len(weak), dtype=np.float32)
    weight[(p <= 0.03) | (p >= 0.97)] = 1.0
    weight[((p > 0.03) & (p <= 0.15)) | ((p >= 0.85) & (p < 0.97))] = 0.6
    weight[((p > 0.15) & (p <= 0.30)) | ((p >= 0.70) & (p < 0.85))] = 0.3
    keep = weight > 0
    if validation_item_ids:
        keep &= ~weak["id1"].isin(validation_item_ids).to_numpy()
        keep &= ~weak["id2"].isin(validation_item_ids).to_numpy()
    return keep


def _prefilter_weak(
    weak: pd.DataFrame,
    validation_item_ids: set[object],
    max_presample_rows: int,
    seed: int,
) -> pd.DataFrame:
    # Local duplicate of the vectorized gate kept intentionally dependency-light
    # for GPU preprocessing.
    keep = _weak_prefilter_mask(weak, validation_item_ids)
    eligible_positions = np.flatnonzero(keep)
    if len(eligible_positions) > max_presample_rows:
        sampled_offsets = np.random.RandomState(seed).choice(
            len(eligible_positions), size=max_presample_rows, replace=False
        )
        eligible_positions = eligible_positions[sampled_offsets]
    return (
        weak.iloc[eligible_positions][["id1", "id2", "target"]]
        .copy()
        .reset_index(drop=True)
    )


def _prefilter_weak_parquet(
    path: Path,
    validation_item_ids: set[object],
    max_presample_rows: int,
    seed: int,
    *,
    batch_size: int = 250_000,
) -> tuple[pd.DataFrame, int]:
    """Select the exact legacy weak sample without materializing all 11M rows."""
    if max_presample_rows <= 0:
        raise ValueError("max_presample_rows must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    columns = ["id1", "id2", "target"]
    parquet = pq.ParquetFile(str(path))
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"weak parquet missing columns: {sorted(missing)}")

    eligible_count = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        eligible_count += int(_weak_prefilter_mask(frame, validation_item_ids).sum())

    if eligible_count > max_presample_rows:
        selected_ordinals = np.random.RandomState(seed).choice(
            eligible_count, size=max_presample_rows, replace=False
        )
    else:
        selected_ordinals = np.arange(eligible_count, dtype=np.int64)

    sorted_order = np.argsort(selected_ordinals)
    selected_sorted = np.asarray(selected_ordinals, dtype=np.int64)[sorted_order]
    pieces: list[pd.DataFrame] = []
    eligible_cursor = 0
    selected_cursor = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        eligible_positions = np.flatnonzero(
            _weak_prefilter_mask(frame, validation_item_ids)
        )
        next_eligible_cursor = eligible_cursor + len(eligible_positions)
        selected_end = int(
            np.searchsorted(selected_sorted, next_eligible_cursor, side="left")
        )
        if selected_end > selected_cursor:
            ordinals = selected_sorted[selected_cursor:selected_end]
            local_offsets = ordinals - eligible_cursor
            raw_positions = eligible_positions[local_offsets]
            piece = frame.iloc[raw_positions][columns].copy()
            piece["_sample_order"] = sorted_order[selected_cursor:selected_end]
            pieces.append(piece)
        selected_cursor = selected_end
        eligible_cursor = next_eligible_cursor

    if eligible_cursor != eligible_count or selected_cursor != len(selected_ordinals):
        raise RuntimeError("weak parquet changed during deterministic sampling")
    if not pieces:
        return pd.DataFrame(columns=columns), int(parquet.metadata.num_rows)
    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values("_sample_order", kind="stable").drop(columns="_sample_order")
    return out.reset_index(drop=True), int(parquet.metadata.num_rows)


def prepare_training_examples(
    human_items_path: Path,
    human_matches_path: Path,
    llm_matches_path: Path,
    full_items_path: Path,
    *,
    weak_presample_rows: int = 500_000,
    weak_final_rows: int = 300_000,
    transitive_cap: int = 1000,
    max_attrs: int = 12,
    max_chars: int = 900,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build leakage-safe outer-train and fixed-validation reranker examples."""
    human_items = pd.read_parquet(
        human_items_path, columns=["id", "name", "attributes", "category"]
    )
    matches = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    matches = attach_pair_category(matches, human_items)
    train_idx, valid_idx = fixed_v1_split(matches)
    outer_train = matches.iloc[train_idx].reset_index(drop=True)
    valid = matches.iloc[valid_idx].reset_index(drop=True)
    valid_items = set(valid["id1"]) | set(valid["id2"])
    train_items = set(outer_train["id1"]) | set(outer_train["id2"])
    overlap = train_items & valid_items
    if overlap:
        raise RuntimeError(f"fixed split has {len(overlap)} overlapping item IDs")

    clean, clean_report = clean_human_pairs(outer_train[["id1", "id2", "target"]])
    augmented, graph_report = augment_transitive_positives(
        clean, max_pairs_per_component=transitive_cap
    )
    clean = attach_pair_category(clean, human_items)
    augmented = attach_pair_category(augmented, human_items)
    importance = learn_attribute_importance(human_items, clean, min_support=20)

    weak, weak_input_rows = _prefilter_weak_parquet(
        llm_matches_path,
        valid_items,
        weak_presample_rows,
        SEED,
    )
    weak, prep_report = prepare_weak_pairs(weak)
    weak, conflict_report = remove_human_conflicts(
        weak, clean[["id1", "id2", "target"]]
    )
    weak_ids = set(weak["id1"]) | set(weak["id2"])
    weak_items = select_items_by_ids(full_items_path, weak_ids)
    weak = attach_pair_category(weak, weak_items)
    weak = sample_weak_training(weak, max_rows=weak_final_rows, seed=SEED)
    final_weak_ids = set(weak["id1"]) | set(weak["id2"])
    weak_items = weak_items[weak_items["id"].isin(final_weak_ids)].reset_index(drop=True)

    human_source = pd.Series(["human"] * len(augmented))
    weak_source = pd.Series(["weak"] * len(weak))
    categories = pd.concat(
        [augmented["category"].reset_index(drop=True), weak["category"].reset_index(drop=True)],
        ignore_index=True,
    )
    source = pd.concat([human_source, weak_source], ignore_index=True)
    weak_weight = np.concatenate(
        [np.ones(len(augmented), dtype=float), weak["weak_weight"].to_numpy(float)]
    )
    sample_weight = _soft_category_weights(categories, source, weak_weight)

    human_pairs = augmented[["id1", "id2", "target", "category"]].copy()
    human_pairs["sample_weight"] = sample_weight[: len(human_pairs)]
    weak_pairs = weak[["id1", "id2", "target", "category"]].copy()
    weak_pairs["sample_weight"] = sample_weight[len(human_pairs) :]
    train_pairs = pd.concat([human_pairs, weak_pairs], ignore_index=True)

    all_items = pd.concat([human_items, weak_items], ignore_index=True).drop_duplicates("id", keep="first")
    train_examples = build_reranker_examples(
        all_items,
        train_pairs,
        importance,
        max_attrs=max_attrs,
        max_chars=max_chars,
    )
    valid_pairs = valid[["id1", "id2", "target", "category"]].copy()
    valid_pairs["sample_weight"] = 1.0
    valid_examples = build_reranker_examples(
        human_items,
        valid_pairs,
        importance,
        max_attrs=max_attrs,
        max_chars=max_chars,
    )

    report = {
        "human_rows": int(len(matches)),
        "outer_train_rows": int(len(outer_train)),
        "validation_rows": int(len(valid)),
        "validation_item_overlap": 0,
        "human_clean_rows": int(len(clean)),
        "human_augmented_rows": int(len(augmented)),
        "weak_input_rows": weak_input_rows,
        "weak_final_rows": int(len(weak)),
        "weak_unique_items": int(len(final_weak_ids)),
        "train_examples": int(len(train_examples)),
        "clean_report": clean_report,
        "graph_report": graph_report,
        "weak_prepare_report": prep_report,
        "weak_conflict_report": conflict_report,
        "attribute_importance": importance,
    }
    return train_examples, valid_examples, report


def _make_dataset_class(Dataset):
    class PairDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.frame = frame.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = self.frame.iloc[index]
            return {
                "text_a": row["text_a"],
                "text_b": row["text_b"],
                "target": float(row["target"]),
                "sample_weight": float(row["sample_weight"]),
                "category": str(row["category"]),
                "id1": row["id1"],
                "id2": row["id2"],
            }

    return PairDataset


def _make_collate(tokenizer, torch, max_length: int):
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [r["text_a"] for r in rows],
            [r["text_b"] for r in rows],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["target"] = torch.tensor([r["target"] for r in rows], dtype=torch.float32)
        encoded["sample_weight"] = torch.tensor(
            [r["sample_weight"] for r in rows], dtype=torch.float32
        )
        encoded["category"] = [r["category"] for r in rows]
        encoded["id1"] = [r["id1"] for r in rows]
        encoded["id2"] = [r["id2"] for r in rows]
        return encoded

    return collate


def _amp_dtype(torch, device: str):
    if device != "cuda":
        return None
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _train_model(
    model,
    tokenizer,
    train_frame: pd.DataFrame,
    *,
    batch_size: int,
    epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    max_length: int,
    gradient_accumulation: int,
    seed: int,
) -> dict[str, Any]:
    torch, DataLoader, Dataset, _, _, get_linear_schedule_with_warmup = _require_torch_transformers()
    _set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    PairDataset = _make_dataset_class(Dataset)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        PairDataset(train_frame),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
        collate_fn=_make_collate(tokenizer, torch, max_length),
        generator=generator,
    )
    effective_batches = max(1, math.ceil(len(loader) * float(epochs)))
    optimizer_steps = max(1, math.ceil(effective_batches / gradient_accumulation))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(optimizer_steps * warmup_ratio),
        num_training_steps=optimizer_steps,
    )
    amp_dtype = _amp_dtype(torch, device)
    use_scaler = device == "cuda" and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except Exception:  # pragma: no cover - older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    seen_batches = 0
    optimizer_step = 0
    while seen_batches < effective_batches:
        for batch in loader:
            if seen_batches >= effective_batches:
                break
            target = batch.pop("target").to(device, non_blocking=True)
            sample_weight = batch.pop("sample_weight").to(device, non_blocking=True)
            batch.pop("category")
            batch.pop("id1")
            batch.pop("id2")
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if amp_dtype is None:
                logits = model(**inputs).logits.squeeze(-1)
                raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
                loss = (raw_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
            else:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(**inputs).logits.squeeze(-1)
                    raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, target, reduction="none"
                    )
                    loss = (raw_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
            scaled_loss = loss / gradient_accumulation
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            losses.append(float(loss.detach().cpu()))
            seen_batches += 1

            if seen_batches % gradient_accumulation == 0 or seen_batches == effective_batches:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1
                if optimizer_step % 100 == 0:
                    recent = sum(losses[-100 * gradient_accumulation :]) / max(
                        1, len(losses[-100 * gradient_accumulation :])
                    )
                    print(
                        f"train step={optimizer_step}/{optimizer_steps} "
                        f"loss={recent:.6f}",
                        flush=True,
                    )
        # DataLoader is recreated logically by Python iteration on the next loop.
    return {
        "device": device,
        "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
        "batches": int(seen_batches),
        "optimizer_steps": int(optimizer_step),
        "mean_loss": float(sum(losses) / max(1, len(losses))),
    }


def _predict(model, tokenizer, frame: pd.DataFrame, *, batch_size: int, max_length: int) -> np.ndarray:
    torch, DataLoader, Dataset, _, _, _ = _require_torch_transformers()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    PairDataset = _make_dataset_class(Dataset)
    loader = DataLoader(
        PairDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
        collate_fn=_make_collate(tokenizer, torch, max_length),
    )
    amp_dtype = _amp_dtype(torch, device)
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("target")
            batch.pop("sample_weight")
            batch.pop("category")
            batch.pop("id1")
            batch.pop("id2")
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if amp_dtype is None:
                logits = model(**inputs).logits.squeeze(-1)
            else:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(**inputs).logits.squeeze(-1)
            scores.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(scores).astype(np.float32) if scores else np.empty(0, dtype=np.float32)


def _evaluate(model, tokenizer, frame: pd.DataFrame, *, batch_size: int, max_length: int) -> tuple[dict[str, Any], np.ndarray]:
    started = time.perf_counter()
    score = _predict(model, tokenizer, frame, batch_size=batch_size, max_length=max_length)
    macro, per_category = macro_average_precision(
        frame["target"].to_numpy(float), score, frame["category"].to_numpy()
    )
    return {
        "macro_average_precision": float(macro),
        "per_category_ap": per_category,
        "prediction_seconds": float(time.perf_counter() - started),
        "rows": int(len(frame)),
    }, score


def _hard_negative_frame(
    model,
    tokenizer,
    train_frame: pd.DataFrame,
    *,
    count: int,
    batch_size: int,
    max_length: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    human = train_frame[train_frame["sample_weight"] > 1.0].copy()
    # Because category normalization can reduce some human weights below one,
    # fall back to hard target rows if the heuristic is too selective.
    if len(human) < count * 2:
        human = train_frame.copy()
    negative = human[human["target"].astype(float) < 0.5].copy().reset_index(drop=True)
    positive = human[human["target"].astype(float) >= 0.5].copy().reset_index(drop=True)
    if not len(negative) or not len(positive):
        return human.iloc[:0].copy(), {"selected_negatives": 0, "selected_positives": 0}
    score = _predict(
        model, tokenizer, negative, batch_size=batch_size, max_length=max_length
    )
    negative["_score"] = score
    hardest = negative.nlargest(min(count, len(negative)), "_score").drop(columns="_score")
    positives = positive.sample(
        n=min(len(hardest), len(positive)), random_state=seed
    )
    stage2 = pd.concat([hardest, positives], ignore_index=True)
    # Human examples are authoritative during hard-negative fine-tuning.
    stage2["sample_weight"] = 1.0
    return stage2, {
        "selected_negatives": int(len(hardest)),
        "selected_positives": int(len(positives)),
        "mean_hard_negative_stage1_score": float(score[np.argsort(score)[-len(hardest):]].mean())
        if len(hardest)
        else 0.0,
    }


def train_reranker(
    human_items_path: Path,
    human_matches_path: Path,
    llm_matches_path: Path,
    full_items_path: Path,
    output_dir: Path,
    *,
    base_model: str = DEFAULT_MODEL,
    weak_presample_rows: int = 500_000,
    weak_final_rows: int = 300_000,
    transitive_cap: int = 1000,
    max_attrs: int = 12,
    max_chars: int = 900,
    max_length: int = 256,
    train_batch_size: int = 128,
    eval_batch_size: int = 256,
    gradient_accumulation: int = 1,
    epochs: float = 1.0,
    hard_epochs: float = 0.30,
    hard_negative_count: int = 50_000,
    learning_rate: float = 3e-5,
    hard_learning_rate: float = 1e-5,
) -> dict[str, Any]:
    started = time.perf_counter()
    _set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_examples, valid_examples, data_report = prepare_training_examples(
        human_items_path,
        human_matches_path,
        llm_matches_path,
        full_items_path,
        weak_presample_rows=weak_presample_rows,
        weak_final_rows=weak_final_rows,
        transitive_cap=transitive_cap,
        max_attrs=max_attrs,
        max_chars=max_chars,
    )
    # Persist safe, derived text datasets to make GPU debugging reproducible.
    train_examples.to_parquet(output_dir / "train_examples.parquet", index=False)
    valid_examples.to_parquet(output_dir / "validation_examples.parquet", index=False)

    torch, _, _, AutoModelForSequenceClassification, AutoTokenizer, _ = _require_torch_transformers()
    if not torch.cuda.is_available():
        raise RuntimeError("v2 reranker full training requires a CUDA GPU")
    print("gpu", torch.cuda.get_device_name(0), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )

    stage1_started = time.perf_counter()
    stage1_train = _train_model(
        model,
        tokenizer,
        train_examples,
        batch_size=train_batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.06,
        max_length=max_length,
        gradient_accumulation=gradient_accumulation,
        seed=SEED,
    )
    stage1_metrics, stage1_score = _evaluate(
        model,
        tokenizer,
        valid_examples,
        batch_size=eval_batch_size,
        max_length=max_length,
    )
    stage1_metrics["train"] = stage1_train
    stage1_metrics["total_seconds"] = float(time.perf_counter() - stage1_started)
    stage1_dir = output_dir / "stage1"
    model.save_pretrained(stage1_dir, safe_serialization=True)
    tokenizer.save_pretrained(stage1_dir)

    hard_frame, hard_report = _hard_negative_frame(
        model,
        tokenizer,
        train_examples,
        count=hard_negative_count,
        batch_size=eval_batch_size,
        max_length=max_length,
        seed=SEED + 31,
    )
    stage2_metrics: dict[str, Any] | None = None
    stage2_score: np.ndarray | None = None
    if len(hard_frame):
        stage2_started = time.perf_counter()
        stage2_train = _train_model(
            model,
            tokenizer,
            hard_frame,
            batch_size=train_batch_size,
            epochs=hard_epochs,
            learning_rate=hard_learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.02,
            max_length=max_length,
            gradient_accumulation=gradient_accumulation,
            seed=SEED + 1,
        )
        stage2_metrics, stage2_score = _evaluate(
            model,
            tokenizer,
            valid_examples,
            batch_size=eval_batch_size,
            max_length=max_length,
        )
        stage2_metrics["train"] = stage2_train
        stage2_metrics["total_seconds"] = float(time.perf_counter() - stage2_started)

    selected_stage = "stage1"
    selected_metrics = stage1_metrics
    selected_score = stage1_score
    if stage2_metrics is not None and stage2_metrics["macro_average_precision"] > stage1_metrics["macro_average_precision"]:
        selected_stage = "stage2"
        selected_metrics = stage2_metrics
        selected_score = stage2_score
        final_dir = output_dir / "model"
        model.save_pretrained(final_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_dir)
    else:
        final_dir = output_dir / "model"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(stage1_dir, final_dir)

    validation_predictions = valid_examples[["id1", "id2", "target", "category"]].copy()
    validation_predictions["stage1_score"] = stage1_score
    if stage2_score is not None:
        validation_predictions["stage2_score"] = stage2_score
    validation_predictions["selected_score"] = selected_score
    validation_predictions.to_parquet(
        output_dir / "validation_predictions.parquet", index=False
    )

    payload = {
        "version": "v2-gpu-reranker",
        "base_model": base_model,
        "seed": SEED,
        "max_length": max_length,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_accumulation": gradient_accumulation,
        "epochs": epochs,
        "hard_epochs": hard_epochs,
        "hard_negative_count": hard_negative_count,
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "hard_negative_report": hard_report,
        "selected_stage": selected_stage,
        "selected_macro_average_precision": float(selected_metrics["macro_average_precision"]),
        "delta_vs_v1": float(selected_metrics["macro_average_precision"] - V1_MACRO_AP),
        "data_report": {k: v for k, v in data_report.items() if k != "attribute_importance"},
        "attribute_importance": data_report["attribute_importance"],
        "total_seconds": float(time.perf_counter() - started),
        "cuda_device": torch.cuda.get_device_name(0),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-items", type=Path, required=True)
    parser.add_argument("--human-matches", type=Path, required=True)
    parser.add_argument("--llm-matches", type=Path, required=True)
    parser.add_argument("--full-items", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--weak-presample-rows", type=int, default=500_000)
    parser.add_argument("--weak-final-rows", type=int, default=300_000)
    parser.add_argument("--transitive-cap", type=int, default=1000)
    parser.add_argument("--max-attrs", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--hard-epochs", type=float, default=0.30)
    parser.add_argument("--hard-negative-count", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--hard-learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    metrics = train_reranker(
        args.human_items,
        args.human_matches,
        args.llm_matches,
        args.full_items,
        args.output_dir,
        base_model=args.base_model,
        weak_presample_rows=args.weak_presample_rows,
        weak_final_rows=args.weak_final_rows,
        transitive_cap=args.transitive_cap,
        max_attrs=args.max_attrs,
        max_chars=args.max_chars,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        epochs=args.epochs,
        hard_epochs=args.hard_epochs,
        hard_negative_count=args.hard_negative_count,
        learning_rate=args.learning_rate,
        hard_learning_rate=args.hard_learning_rate,
    )
    print(json.dumps({
        "selected_stage": metrics["selected_stage"],
        "selected_macro_average_precision": metrics["selected_macro_average_precision"],
        "delta_vs_v1": metrics["delta_vs_v1"],
        "cuda_device": metrics["cuda_device"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
