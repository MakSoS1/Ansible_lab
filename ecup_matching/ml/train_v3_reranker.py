from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .metrics import macro_average_precision
from .train_reranker_v2 import _make_collate, _make_dataset_class, _require_torch_transformers
from .v3_selection import select_best_blend, select_hard_negatives


SEED = 2026
BASE_MODEL = "cointegrated/rubert-tiny2"
V2_MACRO_AP = 0.5010008994958702
PRIORITY_CATEGORIES = {
    "Электроника",
    "Одежда",
    "Обувь",
    "Ювелирные изделия",
    "Галантерея и аксессуары",
    "Мебель",
}


def _select_accelerator(torch_module) -> str:
    if bool(torch_module.cuda.is_available()):
        return "cuda"
    mps = getattr(getattr(torch_module, "backends", object()), "mps", None)
    if mps is not None and bool(mps.is_available()):
        return "mps"
    raise RuntimeError("v3 neural training requires a CUDA or MPS accelerator")


def _set_seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[:0].copy()
    if len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed)


def make_stage2_frame(
    train: pd.DataFrame,
    *,
    human_scores,
    hard_negative_count: int,
    priority_categories: set[str] = PRIORITY_CATEGORIES,
    priority_fraction: float = 0.70,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "source" not in train.columns:
        raise ValueError("train frame must contain source")
    human = train.loc[train["source"].astype(str) == "human"].copy().reset_index(drop=True)
    score = np.asarray(human_scores, dtype=float)
    if len(score) != len(human):
        raise ValueError("human_scores must align exactly to human training rows")
    hard = select_hard_negatives(
        human,
        score,
        count=hard_negative_count,
        priority_categories=priority_categories,
        priority_fraction=priority_fraction,
        seed=seed,
    )
    positives = human.loc[human["target"].astype(float) >= 0.5].copy()
    if positives.empty:
        raise RuntimeError("stage2 requires at least one authoritative human positive")
    priority_pos = positives.loc[positives["category"].astype(str).isin(priority_categories)]
    regular_pos = positives.loc[~positives.index.isin(priority_pos.index)]
    desired_pos = min(max(len(hard), 1), len(positives))
    priority_pos_n = min(math.ceil(desired_pos * priority_fraction), len(priority_pos))
    picked_priority = _sample(priority_pos, priority_pos_n, seed + 1)
    remaining = desired_pos - len(picked_priority)
    picked_regular = _sample(regular_pos, min(remaining, len(regular_pos)), seed + 2)
    remaining -= len(picked_regular)
    if remaining:
        used = set(picked_priority.index) | set(picked_regular.index)
        fill = _sample(positives.loc[~positives.index.isin(used)], remaining, seed + 3)
    else:
        fill = positives.iloc[:0].copy()
    selected_pos = pd.concat([picked_priority, picked_regular, fill], ignore_index=True)
    stage2 = pd.concat([hard, selected_pos], ignore_index=True)
    stage2 = stage2.sample(frac=1.0, random_state=seed + 4).reset_index(drop=True)
    return stage2, {
        "human_rows_scored": int(len(human)),
        "selected_negatives": int(len(hard)),
        "selected_positives": int(len(selected_pos)),
        "priority_negative_rows": int(hard["category"].astype(str).isin(priority_categories).sum()),
        "rows": int(len(stage2)),
    }


def category_aware_blend(
    frame: pd.DataFrame,
    *,
    structured_col: str,
    neural_col: str,
    allowed_categories: set[str],
    alphas: Iterable[float] = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0),
) -> dict[str, Any]:
    required = {"target", "category", structured_col, neural_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"blend frame missing columns: {sorted(missing)}")
    structured = frame[structured_col].to_numpy(float)
    neural = frame[neural_col].to_numpy(float)
    target = frame["target"].to_numpy(float)
    categories = frame["category"].astype(str).to_numpy()
    base_macro, base_per_category = macro_average_precision(target, structured, categories)
    output = structured.copy()
    category_alphas: dict[str, float] = {}
    category_reports: dict[str, Any] = {}
    for category in sorted(set(categories)):
        mask = categories == category
        if category not in allowed_categories:
            category_alphas[category] = 0.0
            continue
        result = select_best_blend(
            structured[mask],
            neural[mask],
            target[mask],
            categories[mask],
            alphas=tuple(float(value) for value in alphas),
        )
        alpha = float(result["alpha_neural"])
        category_alphas[category] = alpha
        output[mask] = np.asarray(result["scores"], dtype=float)
        category_reports[category] = {
            "alpha_neural": alpha,
            "structured_ap": float(result["structured_macro_average_precision"]),
            "selected_ap": float(result["macro_average_precision"]),
        }
    macro, per_category = macro_average_precision(target, output, categories)
    return {
        "macro_average_precision": float(macro),
        "per_category_ap": per_category,
        "structured_macro_average_precision": float(base_macro),
        "structured_per_category_ap": base_per_category,
        "category_alphas": category_alphas,
        "category_reports": category_reports,
        "scores": np.clip(output, 0.0, 1.0),
    }


def _train_steps(
    model,
    tokenizer,
    frame: pd.DataFrame,
    *,
    torch,
    device,
    max_length: int,
    batch_size: int,
    learning_rate: float,
    max_steps: int,
    seed: int,
) -> dict[str, float]:
    PairDataset = _make_dataset_class(torch.utils.data.Dataset)
    DataLoader = torch.utils.data.DataLoader
    loader = DataLoader(
        PairDataset(frame),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=_make_collate(tokenizer, torch, max_length),
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    model.train()
    model.to(device)
    started = time.perf_counter()
    steps = 0
    losses: list[float] = []
    while steps < max_steps:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)
            weights = batch["sample_weight"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            per_row = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            loss = (per_row * weights).sum() / weights.sum().clamp_min(1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            steps += 1
            if steps % 100 == 0 or steps == 1:
                print(
                    f"train_step={steps}/{max_steps} loss={losses[-1]:.6f} device={device.type}",
                    flush=True,
                )
            if steps >= max_steps:
                break
    return {
        "steps": int(steps),
        "seconds": float(time.perf_counter() - started),
        "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        "last_loss": float(losses[-1]) if losses else float("nan"),
    }


def _predict(
    model,
    tokenizer,
    frame: pd.DataFrame,
    *,
    torch,
    device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    PairDataset = _make_dataset_class(torch.utils.data.Dataset)
    loader = torch.utils.data.DataLoader(
        PairDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=_make_collate(tokenizer, torch, max_length),
    )
    model.eval()
    model.to(device)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits.squeeze(-1)
            chunks.append(torch.sigmoid(logits).float().cpu().numpy())
    out = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("reranker prediction contains NaN/Inf")
    return out.astype(np.float32, copy=False)


def _align_structured_validation(
    valid: pd.DataFrame, structured_path: Path, structured_col: str
) -> pd.DataFrame:
    structured = pd.read_parquet(structured_path)
    required = {"id1", "id2", "target", "category", structured_col}
    missing = required - set(structured.columns)
    if missing:
        raise ValueError(f"structured validation missing columns: {sorted(missing)}")
    left = valid[["id1", "id2", "target", "category"]].copy()
    left["_row"] = np.arange(len(left))
    joined = left.merge(
        structured[["id1", "id2", "target", "category", structured_col]],
        on=["id1", "id2"],
        how="left",
        suffixes=("", "_structured"),
        validate="one_to_one",
    ).sort_values("_row")
    if len(joined) != len(valid) or joined[structured_col].isna().any():
        raise RuntimeError("structured validation predictions do not align to fixed validation")
    if not np.allclose(joined["target"].to_numpy(float), joined["target_structured"].to_numpy(float)):
        raise RuntimeError("structured validation target mismatch")
    if not (joined["category"].astype(str).to_numpy() == joined["category_structured"].astype(str).to_numpy()).all():
        raise RuntimeError("structured validation category mismatch")
    return joined[["id1", "id2", "target", "category", structured_col]].reset_index(drop=True)


def _evaluate_neural(valid: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    macro, per_category = macro_average_precision(
        valid["target"].to_numpy(), scores, valid["category"].astype(str).to_numpy()
    )
    return {"macro_average_precision": float(macro), "per_category_ap": per_category}


def train_v3(
    *,
    train_path: Path,
    validation_path: Path,
    structured_validation_path: Path,
    output_dir: Path,
    model_name: str = BASE_MODEL,
    structured_col: str = "v2b-weak-curriculum",
    max_length: int = 160,
    train_batch_size: int = 24,
    eval_batch_size: int = 64,
    stage1_steps: int = 2400,
    stage2_steps: int = 450,
    stage1_lr: float = 2e-5,
    stage2_lr: float = 8e-6,
    hard_negative_count: int = 12_000,
) -> dict[str, Any]:
    (
        torch,
        _,
        _,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        _,
    ) = _require_torch_transformers()
    _set_seed(torch, SEED)
    accelerator = _select_accelerator(torch)
    device = torch.device(accelerator)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(train_path)
    valid = pd.read_parquet(validation_path)
    required = {"id1", "id2", "target", "category", "sample_weight", "text_a", "text_b", "source"}
    for label, frame in (("train", train), ("validation", valid)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} examples missing columns: {sorted(missing)}")
    valid_ids = set(valid["id1"]) | set(valid["id2"])
    train_ids = set(train["id1"]) | set(train["id2"])
    overlap = train_ids & valid_ids
    if overlap:
        raise RuntimeError(f"v3 prepared data has {len(overlap)} validation item overlaps")

    aligned = _align_structured_validation(valid, structured_validation_path, structured_col)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    started = time.perf_counter()
    stage1_train = _train_steps(
        model,
        tokenizer,
        train,
        torch=torch,
        device=device,
        max_length=max_length,
        batch_size=train_batch_size,
        learning_rate=stage1_lr,
        max_steps=stage1_steps,
        seed=SEED,
    )
    stage1_scores = _predict(
        model,
        tokenizer,
        valid,
        torch=torch,
        device=device,
        max_length=max_length,
        batch_size=eval_batch_size,
    )
    stage1_metric = _evaluate_neural(valid, stage1_scores)
    stage1_dir = output_dir / "stage1-model"
    model.save_pretrained(stage1_dir, safe_serialization=True)
    tokenizer.save_pretrained(stage1_dir)

    human = train.loc[train["source"].astype(str) == "human"].reset_index(drop=True)
    human_scores = _predict(
        model,
        tokenizer,
        human,
        torch=torch,
        device=device,
        max_length=max_length,
        batch_size=eval_batch_size,
    )
    stage2_frame, mining_report = make_stage2_frame(
        train,
        human_scores=human_scores,
        hard_negative_count=hard_negative_count,
        priority_categories=PRIORITY_CATEGORIES,
        priority_fraction=0.70,
        seed=SEED,
    )
    stage2_frame = stage2_frame.copy()
    stage2_frame["sample_weight"] = 1.0
    stage2_train = _train_steps(
        model,
        tokenizer,
        stage2_frame,
        torch=torch,
        device=device,
        max_length=max_length,
        batch_size=train_batch_size,
        learning_rate=stage2_lr,
        max_steps=stage2_steps,
        seed=SEED + 10,
    )
    stage2_scores = _predict(
        model,
        tokenizer,
        valid,
        torch=torch,
        device=device,
        max_length=max_length,
        batch_size=eval_batch_size,
    )
    stage2_metric = _evaluate_neural(valid, stage2_scores)

    validation = aligned.copy()
    validation["neural_stage1"] = stage1_scores
    validation["neural_stage2"] = stage2_scores
    global_results: dict[str, Any] = {}
    category_results: dict[str, Any] = {}
    candidates: list[tuple[str, str, float, np.ndarray, dict[str, float]]] = []
    for stage_name, score in (("stage1", stage1_scores), ("stage2", stage2_scores)):
        global_blend = select_best_blend(
            aligned[structured_col].to_numpy(float),
            score,
            aligned["target"].to_numpy(float),
            aligned["category"].astype(str).to_numpy(),
            alphas=(0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0),
        )
        global_results[stage_name] = {
            key: value for key, value in global_blend.items() if key != "scores"
        }
        candidates.append(
            (
                f"{stage_name}-global",
                stage_name,
                float(global_blend["macro_average_precision"]),
                np.asarray(global_blend["scores"], dtype=float),
                {"__global__": float(global_blend["alpha_neural"])},
            )
        )
        tmp = aligned.copy()
        tmp["neural"] = score
        category_blend = category_aware_blend(
            tmp,
            structured_col=structured_col,
            neural_col="neural",
            allowed_categories=PRIORITY_CATEGORIES,
        )
        category_results[stage_name] = {
            key: value for key, value in category_blend.items() if key != "scores"
        }
        candidates.append(
            (
                f"{stage_name}-priority-category",
                stage_name,
                float(category_blend["macro_average_precision"]),
                np.asarray(category_blend["scores"], dtype=float),
                {str(k): float(v) for k, v in category_blend["category_alphas"].items()},
            )
        )

    selected_name, selected_stage, selected_macro, selected_scores, selected_alphas = max(
        candidates, key=lambda item: item[2]
    )
    selected_neural_scores = stage1_scores if selected_stage == "stage1" else stage2_scores
    selected_dir = output_dir / "model"
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    if selected_stage == "stage1":
        shutil.copytree(stage1_dir, selected_dir)
    else:
        model.save_pretrained(selected_dir, safe_serialization=True)
        tokenizer.save_pretrained(selected_dir)

    validation["selected_neural"] = selected_neural_scores
    validation["selected_score"] = selected_scores
    validation.to_parquet(output_dir / "validation_predictions.parquet", index=False)

    selected_macro_exact, selected_per_category = macro_average_precision(
        validation["target"].to_numpy(),
        selected_scores,
        validation["category"].astype(str).to_numpy(),
    )
    payload: dict[str, Any] = {
        "version": "v3-compact-reranker",
        "seed": SEED,
        "accelerator": accelerator,
        "model_name": model_name,
        "max_length": int(max_length),
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "validation_item_overlap": 0,
        "stage1_training": stage1_train,
        "stage1_validation": stage1_metric,
        "hard_negative_mining": mining_report,
        "stage2_training": stage2_train,
        "stage2_validation": stage2_metric,
        "global_blends": global_results,
        "priority_category_blends": category_results,
        "selected_candidate": selected_name,
        "selected_model_stage": selected_stage,
        "selected_macro_average_precision": float(selected_macro_exact),
        "selected_per_category_ap": selected_per_category,
        "selected_category_alphas": selected_alphas,
        "v2_anchor_macro_average_precision": V2_MACRO_AP,
        "delta_vs_v2": float(selected_macro_exact - V2_MACRO_AP),
        "accepted_as_improvement": bool(selected_macro_exact > V2_MACRO_AP),
        "total_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "version": "v3-compact-reranker",
        "base_model": model_name,
        "selected_model_stage": selected_stage,
        "selected_candidate": selected_name,
        "validation_macro_ap": float(selected_macro_exact),
        "max_length": int(max_length),
        "max_attrs": 10,
        "max_chars": 700,
        "structured_column": structured_col,
        "priority_categories": sorted(PRIORITY_CATEGORIES),
        "category_alphas": selected_alphas,
        "seed": SEED,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--structured-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--train-batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--stage1-steps", type=int, default=2400)
    parser.add_argument("--stage2-steps", type=int, default=450)
    parser.add_argument("--stage1-lr", type=float, default=2e-5)
    parser.add_argument("--stage2-lr", type=float, default=8e-6)
    parser.add_argument("--hard-negative-count", type=int, default=12000)
    args = parser.parse_args()
    train_v3(
        train_path=args.train,
        validation_path=args.validation,
        structured_validation_path=args.structured_validation,
        output_dir=args.output_dir,
        model_name=args.model,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        hard_negative_count=args.hard_negative_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
