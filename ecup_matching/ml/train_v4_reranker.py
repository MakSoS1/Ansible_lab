from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .metrics import macro_average_precision
from .train_reranker_v2 import (
    _evaluate,
    _predict,
    _require_torch_transformers,
    _soft_category_weights,
    _train_model,
    prepare_training_examples,
)
from .train_v2_structured import train_structured_ablation
from .v3_selection import select_best_blend, select_hard_negatives
from .v4_curriculum import (
    assert_item_disjoint,
    build_hard_replay_curriculum,
    build_human_curriculum,
    build_weak_curriculum,
)


SEED = 2026
V3_MACRO_AP = 0.5254642645846543
DEFAULT_MODEL = "ai-forever/ruBert-base"
DEFAULT_MODEL_REVISION = "43be4261797042e172adf7476c558734f3cbb2a0"
STRUCTURED_COLUMN = "v2b-weak-curriculum"
EXPECTED_VALIDATION_ROWS = 73_131
ALPHA_GRID = tuple(round(value / 20.0, 2) for value in range(21))
PRIORITY_CATEGORIES = {
    "Электроника",
    "Одежда",
    "Обувь",
    "Ювелирные изделия",
    "Галантерея и аксессуары",
    "Мебель",
}


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def select_v4_candidate(
    stages: dict[str, dict[str, Any]],
    *,
    baseline_macro_average_precision: float = V3_MACRO_AP,
) -> tuple[str, float] | None:
    """Return the best stage only when it strictly beats retained v3."""
    baseline = _finite(baseline_macro_average_precision, "baseline Macro AP")
    eligible: list[tuple[str, float]] = []
    for stage_name, report in stages.items():
        if "selected_macro_average_precision" not in report:
            raise ValueError(f"stage {stage_name} has no selected Macro AP")
        score = _finite(
            report["selected_macro_average_precision"],
            f"{stage_name} selected Macro AP",
        )
        if score > baseline:
            eligible.append((str(stage_name), score))
    if not eligible:
        return None
    stage_order = {"v4a": 0, "v4b": 1, "v4c": 2}
    return max(eligible, key=lambda item: (item[1], -stage_order.get(item[0], 99)))


def shrink_category_alphas(
    raw_category_alphas: dict[str, float],
    *,
    global_alpha: float,
    category_support: dict[str, int],
    prior_strength: int = 1_000,
) -> dict[str, float]:
    """Shrink category-specific blend weights toward the global optimum."""
    global_value = float(global_alpha)
    if not 0.0 <= global_value <= 1.0:
        raise ValueError("global_alpha must be in [0,1]")
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative")
    if set(raw_category_alphas) != set(category_support):
        raise ValueError("category alpha/support keys must match exactly")
    result: dict[str, float] = {}
    for category in sorted(raw_category_alphas):
        raw = float(raw_category_alphas[category])
        support = int(category_support[category])
        if not 0.0 <= raw <= 1.0:
            raise ValueError("raw category alpha must be in [0,1]")
        if support < 0:
            raise ValueError("category support must be non-negative")
        if prior_strength == 0:
            value = raw
        else:
            weight = support / float(support + prior_strength)
            value = global_value + weight * (raw - global_value)
        result[category] = float(np.clip(value, 0.0, 1.0))
    return result


def build_v4_metrics_payload(
    *,
    stages: dict[str, dict[str, Any]],
    selected_stage: str,
    selected_macro_average_precision: float,
    validation_rows: int,
    validation_item_overlap: int,
    base_model: str,
    base_model_revision: str,
    cuda_device: str,
    train_rows_human: int,
    train_rows_weak: int,
    total_seconds: float,
    baseline_macro_average_precision: float = V3_MACRO_AP,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if int(validation_rows) != EXPECTED_VALIDATION_ROWS:
        raise ValueError("v4 comparable validation must contain exactly 73,131 rows")
    if int(validation_item_overlap) != 0:
        raise ValueError("v4 validation item overlap must be zero")
    if selected_stage not in stages:
        raise ValueError("selected_stage must exist in stages")
    selected = stages[selected_stage]
    per_category = selected.get("selected_per_category_ap")
    if not isinstance(per_category, dict) or len(per_category) != 20:
        raise ValueError("selected stage must record exactly 20 category AP values")
    revision = str(base_model_revision).strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("base_model_revision must be an exact 40-character git SHA")
    device = str(cuda_device)
    if "NVIDIA" not in device.upper():
        raise ValueError("retained v4 training evidence requires an NVIDIA CUDA device")
    selected_macro = _finite(selected_macro_average_precision, "selected Macro AP")
    baseline = _finite(baseline_macro_average_precision, "baseline Macro AP")
    seconds = _finite(total_seconds, "total_seconds")
    if seconds < 0:
        raise ValueError("total_seconds must be non-negative")
    payload: dict[str, Any] = {
        "version": "v4-strong-reranker",
        "seed": SEED,
        "baseline_version": "v3",
        "baseline_macro_average_precision": baseline,
        "selected_stage": str(selected_stage),
        "selected_macro_average_precision": selected_macro,
        "delta_vs_v3": float(selected_macro - baseline),
        "accepted_as_improvement": bool(selected_macro > baseline),
        "validation_rows": int(validation_rows),
        "validation_item_overlap": 0,
        "base_model": str(base_model),
        "base_model_revision": revision.lower(),
        "cuda_device": device,
        "train_rows_human": int(train_rows_human),
        "train_rows_weak": int(train_rows_weak),
        "total_seconds": seconds,
        "stages": stages,
    }
    if extra:
        payload.update(extra)
    return payload


def _reweight_curriculum(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    source = out["source"].astype(str).reset_index(drop=True)
    weak_weight = np.ones(len(out), dtype=float)
    weak_mask = source.to_numpy() == "weak"
    if weak_mask.any():
        # build_weak_curriculum stores the confidence tier in sample_weight.
        weak_weight[weak_mask] = out.loc[weak_mask, "sample_weight"].to_numpy(float)
    out["sample_weight"] = _soft_category_weights(
        out["category"].astype(str).reset_index(drop=True),
        source,
        weak_weight,
    )
    return out


def _prepare_v4_curricula(
    *,
    human_items_path: Path,
    human_matches_path: Path,
    llm_matches_path: Path,
    full_items_path: Path,
    weak_presample_rows: int,
    weak_final_rows: int,
    max_attrs: int,
    max_chars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    combined, valid, report = prepare_training_examples(
        human_items_path,
        human_matches_path,
        llm_matches_path,
        full_items_path,
        weak_presample_rows=weak_presample_rows,
        weak_final_rows=weak_final_rows,
        transitive_cap=1_000,
        max_attrs=max_attrs,
        max_chars=max_chars,
    )
    human_rows = int(report["human_augmented_rows"])
    if human_rows <= 0 or human_rows > len(combined):
        raise RuntimeError("invalid authoritative-human row boundary from preparation")
    combined = combined.copy().reset_index(drop=True)
    combined["source"] = "weak"
    combined.loc[: human_rows - 1, "source"] = "human"
    valid = valid.copy().reset_index(drop=True)
    valid["source"] = "human"

    human = build_human_curriculum(combined.iloc[:human_rows].copy())
    human = _reweight_curriculum(human)
    weak_only = combined.iloc[human_rows:].copy().reset_index(drop=True)
    if len(weak_only) != int(report["weak_final_rows"]):
        raise RuntimeError("prepared weak row boundary does not match preparation report")
    mixed = build_weak_curriculum(
        human,
        weak_only,
        valid,
        max_weak_rows=weak_final_rows,
        seed=SEED,
    )
    mixed = _reweight_curriculum(mixed)
    assert_item_disjoint(human, valid)
    assert_item_disjoint(mixed, valid)
    if len(valid) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError(
            f"fixed validation changed: {len(valid)} rows instead of {EXPECTED_VALIDATION_ROWS}"
        )
    report = dict(report)
    report["v4_human_rows"] = int(len(human))
    report["v4_weak_rows"] = int((mixed["source"].astype(str) == "weak").sum())
    report["v4_mixed_rows"] = int(len(mixed))
    return human, mixed, valid, report


def _align_structured_validation(
    valid: pd.DataFrame,
    structured_path: Path,
    structured_col: str = STRUCTURED_COLUMN,
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
    if not np.allclose(
        joined["target"].to_numpy(float), joined["target_structured"].to_numpy(float)
    ):
        raise RuntimeError("structured validation target mismatch")
    if not (
        joined["category"].astype(str).to_numpy()
        == joined["category_structured"].astype(str).to_numpy()
    ).all():
        raise RuntimeError("structured validation category mismatch")
    return joined[["id1", "id2", "target", "category", structured_col]].reset_index(
        drop=True
    )


def _category_shrunk_blend(
    valid: pd.DataFrame,
    structured: np.ndarray,
    neural: np.ndarray,
    *,
    global_alpha: float,
    alphas: Iterable[float] = ALPHA_GRID,
    prior_strength: int = 1_000,
) -> dict[str, Any]:
    target = valid["target"].to_numpy(float)
    category = valid["category"].astype(str).to_numpy()
    raw_alphas: dict[str, float] = {}
    support: dict[str, int] = {}
    raw_reports: dict[str, dict[str, float]] = {}
    for name in sorted(set(category)):
        mask = category == name
        result = select_best_blend(
            structured[mask],
            neural[mask],
            target[mask],
            category[mask],
            alphas=tuple(float(value) for value in alphas),
        )
        raw_alphas[name] = float(result["alpha_neural"])
        support[name] = int(mask.sum())
        raw_reports[name] = {
            "raw_alpha_neural": float(result["alpha_neural"]),
            "raw_selected_ap": float(result["macro_average_precision"]),
            "structured_ap": float(result["structured_macro_average_precision"]),
        }
    shrunk = shrink_category_alphas(
        raw_alphas,
        global_alpha=global_alpha,
        category_support=support,
        prior_strength=prior_strength,
    )
    score = np.asarray(structured, dtype=float).copy()
    for name, alpha in shrunk.items():
        mask = category == name
        score[mask] = (1.0 - alpha) * structured[mask] + alpha * neural[mask]
    score = np.clip(score, 0.0, 1.0)
    macro, per_category = macro_average_precision(target, score, category)
    return {
        "macro_average_precision": float(macro),
        "per_category_ap": per_category,
        "category_alphas": shrunk,
        "raw_category_alphas": raw_alphas,
        "category_support": support,
        "category_reports": raw_reports,
        "scores": score,
    }


def _evaluate_stage(
    *,
    stage_name: str,
    valid: pd.DataFrame,
    structured_scores: np.ndarray,
    neural_scores: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    target = valid["target"].to_numpy(float)
    category = valid["category"].astype(str).to_numpy()
    neural_macro, neural_per_category = macro_average_precision(
        target, neural_scores, category
    )
    global_blend = select_best_blend(
        structured_scores,
        neural_scores,
        target,
        category,
        alphas=ALPHA_GRID,
    )
    shrunk = _category_shrunk_blend(
        valid,
        structured_scores,
        neural_scores,
        global_alpha=float(global_blend["alpha_neural"]),
        alphas=ALPHA_GRID,
    )
    candidates = [
        (
            "global",
            float(global_blend["macro_average_precision"]),
            np.asarray(global_blend["scores"], dtype=float),
            global_blend["per_category_ap"],
        ),
        (
            "shrunk-category",
            float(shrunk["macro_average_precision"]),
            np.asarray(shrunk["scores"], dtype=float),
            shrunk["per_category_ap"],
        ),
    ]
    selected_name, selected_macro, selected_scores, selected_per_category = max(
        candidates, key=lambda row: (row[1], row[0] == "global")
    )
    report: dict[str, Any] = {
        "stage": stage_name,
        "neural_macro_average_precision": float(neural_macro),
        "neural_per_category_ap": neural_per_category,
        "global_macro_average_precision": float(global_blend["macro_average_precision"]),
        "global_per_category_ap": global_blend["per_category_ap"],
        "global_alpha_neural": float(global_blend["alpha_neural"]),
        "shrunk_category_macro_average_precision": float(shrunk["macro_average_precision"]),
        "shrunk_category_per_category_ap": shrunk["per_category_ap"],
        "shrunk_category_alphas": shrunk["category_alphas"],
        "raw_category_alphas": shrunk["raw_category_alphas"],
        "category_support": shrunk["category_support"],
        "selected_blend": selected_name,
        "selected_macro_average_precision": float(selected_macro),
        "selected_per_category_ap": selected_per_category,
    }
    return report, np.clip(selected_scores, 0.0, 1.0)


def _load_local_model(
    model_path: str,
    *,
    gradient_checkpointing: bool,
):
    (
        torch,
        _,
        _,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        _,
    ) = _require_torch_transformers()
    if not torch.cuda.is_available():
        raise RuntimeError("v4 production training requires NVIDIA CUDA")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return torch, tokenizer, model


def _verify_model_revision(model_path: str, revision: str) -> None:
    path = Path(model_path)
    marker = path / "REVISION"
    if marker.exists():
        actual = marker.read_text(encoding="utf-8").strip().lower()
        if actual != revision.strip().lower():
            raise RuntimeError(
                f"trusted base-model revision mismatch: marker={actual} requested={revision}"
            )


def _save_checkpoint(model, tokenizer, destination: Path, revision: str) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    (destination / "BASE_MODEL_REVISION").write_text(
        revision.strip().lower() + "\n", encoding="utf-8"
    )


def _mine_v4c_frame(
    *,
    model,
    tokenizer,
    parent: pd.DataFrame,
    batch_size: int,
    max_length: int,
    hard_negative_count: int,
    mining_weak_negative_rows: int,
    replay_total_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    human_neg = parent.loc[
        (parent["source"].astype(str) == "human")
        & (parent["target"].astype(float) < 0.5)
    ].copy()
    weak_neg = parent.loc[
        (parent["source"].astype(str) == "weak")
        & (parent["target"].astype(float) < 0.5)
    ].copy()
    if len(weak_neg) > mining_weak_negative_rows:
        weak_neg = weak_neg.sample(
            n=mining_weak_negative_rows, random_state=SEED + 31
        )
    mining_pool = pd.concat([human_neg, weak_neg], ignore_index=True)
    if mining_pool.empty:
        raise RuntimeError("v4c mining pool has no negative examples")
    scores = _predict(
        model,
        tokenizer,
        mining_pool,
        batch_size=batch_size,
        max_length=max_length,
    )
    hard = select_hard_negatives(
        mining_pool,
        scores,
        count=min(hard_negative_count, len(mining_pool)),
        priority_categories=PRIORITY_CATEGORIES,
        priority_fraction=0.70,
        seed=SEED + 32,
    )
    positives = parent.loc[parent["target"].astype(float) >= 0.5].copy()
    if positives.empty:
        raise RuntimeError("v4c requires positive replay examples")
    total_rows = min(
        int(replay_total_rows),
        4 * len(hard),
        4 * len(positives),
        2 * len(parent),
    )
    total_rows -= total_rows % 4
    if total_rows < 4:
        raise RuntimeError("v4c cannot build a non-empty 25/25/50 replay curriculum")
    frame = build_hard_replay_curriculum(
        parent,
        hard,
        positives,
        total_rows=total_rows,
        seed=SEED + 33,
    )
    report = {
        "human_negative_pool_rows": int(len(human_neg)),
        "weak_negative_pool_rows": int(len(weak_neg)),
        "mining_pool_rows": int(len(mining_pool)),
        "selected_hard_negatives": int(len(hard)),
        "priority_hard_negatives": int(
            hard["category"].astype(str).isin(PRIORITY_CATEGORIES).sum()
        ),
        "replay_rows": int(len(frame)),
        "hard_negative_rows": int((frame["curriculum_role"] == "hard_negative").sum()),
        "positive_rows": int((frame["curriculum_role"] == "positive").sum()),
        "ordinary_replay_rows": int((frame["curriculum_role"] == "replay").sum()),
        "mean_mining_negative_score": float(np.mean(scores)),
        "max_mining_negative_score": float(np.max(scores)),
    }
    return frame, report


def benchmark_v4(
    *,
    human_items_path: Path,
    human_matches_path: Path,
    output_dir: Path,
    model_name: str,
    base_model_revision: str,
    max_length: int,
    train_batch_size: int,
    gradient_accumulation: int,
    gradient_checkpointing: bool,
    benchmark_rows: int = 2_048,
    benchmark_epochs: float = 0.20,
    max_attrs: int = 10,
    max_chars: int = 700,
) -> dict[str, Any]:
    from ecup_matching.v3_prepare import prepare_v3_human_only_data

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = output_dir / "benchmark-prepared"
    prepare_v3_human_only_data(
        human_items_path=human_items_path,
        human_matches_path=human_matches_path,
        output_dir=prepared_dir,
        max_train_rows=benchmark_rows,
        max_attrs=max_attrs,
        max_chars=max_chars,
    )
    frame = pd.read_parquet(prepared_dir / "train_examples.parquet")
    _verify_model_revision(model_name, base_model_revision)
    torch, tokenizer, model = _load_local_model(
        model_name, gradient_checkpointing=gradient_checkpointing
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_info = _train_model(
        model,
        tokenizer,
        frame,
        batch_size=train_batch_size,
        epochs=benchmark_epochs,
        learning_rate=1.5e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        max_length=max_length,
        gradient_accumulation=gradient_accumulation,
        seed=SEED,
    )
    seconds = time.perf_counter() - started
    examples_seen = int(train_info["batches"]) * int(train_batch_size)
    payload = {
        "version": "v4-cuda-benchmark",
        "base_model": DEFAULT_MODEL,
        "base_model_revision": base_model_revision.lower(),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": str(torch.__version__),
        "max_length": int(max_length),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation": int(gradient_accumulation),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "benchmark_rows": int(len(frame)),
        "examples_seen": examples_seen,
        "seconds": float(seconds),
        "examples_per_second": float(examples_seen / max(seconds, 1e-9)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "training": train_info,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.rmtree(prepared_dir, ignore_errors=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def train_v4(
    *,
    human_items_path: Path,
    human_matches_path: Path,
    llm_matches_path: Path,
    full_items_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    base_model_revision: str = DEFAULT_MODEL_REVISION,
    max_length: int = 256,
    train_batch_size: int = 2,
    eval_batch_size: int = 16,
    gradient_accumulation: int = 16,
    gradient_checkpointing: bool = False,
    v4a_epochs: float = 1.0,
    v4b_epochs: float = 0.35,
    v4c_epochs: float = 1.0,
    v4a_lr: float = 1.5e-5,
    v4b_lr: float = 6e-6,
    v4c_lr: float = 3e-6,
    weak_presample_rows: int = 1_000_000,
    weak_final_rows: int = 600_000,
    hard_negative_count: int = 20_000,
    mining_weak_negative_rows: int = 120_000,
    replay_total_rows: int = 80_000,
    max_attrs: int = 10,
    max_chars: int = 700,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    _verify_model_revision(model_name, base_model_revision)
    torch, tokenizer, model = _load_local_model(
        model_name, gradient_checkpointing=gradient_checkpointing
    )
    cuda_device = torch.cuda.get_device_name(0)

    # Reproduce the retained structured anchor from raw competition data so the
    # isolated GPU runner never needs private-HF credentials.
    structured_dir = output_dir / "structured-anchor"
    structured_metrics = train_structured_ablation(
        human_items_path,
        human_matches_path,
        llm_matches_path,
        full_items_path,
        structured_dir,
        weak_presample_rows=500_000,
        weak_final_rows=300_000,
        transitive_cap=2_000,
        chunk_size=25_000,
        max_iter=350,
        hard_negative_boost=3.0,
    )
    structured_validation_path = structured_dir / "validation_predictions.parquet"
    if STRUCTURED_COLUMN not in pd.read_parquet(
        structured_validation_path, columns=None
    ).columns:
        raise RuntimeError("reproduced structured validation is missing v2b scores")

    human, mixed, valid, data_report = _prepare_v4_curricula(
        human_items_path=human_items_path,
        human_matches_path=human_matches_path,
        llm_matches_path=llm_matches_path,
        full_items_path=full_items_path,
        weak_presample_rows=weak_presample_rows,
        weak_final_rows=weak_final_rows,
        max_attrs=max_attrs,
        max_chars=max_chars,
    )
    aligned = _align_structured_validation(valid, structured_validation_path)
    structured_scores = aligned[STRUCTURED_COLUMN].to_numpy(float)

    stages_runtime: dict[str, Any] = {}
    stage_reports: dict[str, dict[str, Any]] = {}
    stage_selected_scores: dict[str, np.ndarray] = {}
    stage_neural_scores: dict[str, np.ndarray] = {}
    stage_dirs: dict[str, Path] = {}

    # v4a: all authoritative human rows.
    torch.cuda.reset_peak_memory_stats()
    stage_started = time.perf_counter()
    train_a = _train_model(
        model,
        tokenizer,
        human,
        batch_size=train_batch_size,
        epochs=v4a_epochs,
        learning_rate=v4a_lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        max_length=max_length,
        gradient_accumulation=gradient_accumulation,
        seed=SEED,
    )
    eval_a, neural_a = _evaluate(
        model,
        tokenizer,
        valid,
        batch_size=eval_batch_size,
        max_length=max_length,
    )
    report_a, selected_a = _evaluate_stage(
        stage_name="v4a",
        valid=aligned,
        structured_scores=structured_scores,
        neural_scores=neural_a,
    )
    report_a["training"] = train_a
    report_a["neural_validation"] = eval_a
    report_a["seconds"] = float(time.perf_counter() - stage_started)
    report_a["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    report_a["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    dir_a = output_dir / "v4a-model"
    _save_checkpoint(model, tokenizer, dir_a, base_model_revision)
    stage_dirs["v4a"] = dir_a
    stage_reports["v4a"] = report_a
    stage_selected_scores["v4a"] = selected_a
    stage_neural_scores["v4a"] = neural_a
    stages_runtime["v4a"] = train_a

    # v4b: continue with high-confidence weak supervision while keeping all human rows.
    torch.cuda.reset_peak_memory_stats()
    stage_started = time.perf_counter()
    train_b = _train_model(
        model,
        tokenizer,
        mixed,
        batch_size=train_batch_size,
        epochs=v4b_epochs,
        learning_rate=v4b_lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        max_length=max_length,
        gradient_accumulation=gradient_accumulation,
        seed=SEED + 10,
    )
    eval_b, neural_b = _evaluate(
        model,
        tokenizer,
        valid,
        batch_size=eval_batch_size,
        max_length=max_length,
    )
    report_b, selected_b = _evaluate_stage(
        stage_name="v4b",
        valid=aligned,
        structured_scores=structured_scores,
        neural_scores=neural_b,
    )
    report_b["training"] = train_b
    report_b["neural_validation"] = eval_b
    report_b["seconds"] = float(time.perf_counter() - stage_started)
    report_b["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    report_b["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    dir_b = output_dir / "v4b-model"
    _save_checkpoint(model, tokenizer, dir_b, base_model_revision)
    stage_dirs["v4b"] = dir_b
    stage_reports["v4b"] = report_b
    stage_selected_scores["v4b"] = selected_b
    stage_neural_scores["v4b"] = neural_b
    stages_runtime["v4b"] = train_b

    parent_stage = max(
        ("v4a", "v4b"),
        key=lambda name: float(stage_reports[name]["selected_macro_average_precision"]),
    )
    _, tokenizer_c, model_c = _load_local_model(
        str(stage_dirs[parent_stage]),
        gradient_checkpointing=gradient_checkpointing,
    )
    parent_frame = human if parent_stage == "v4a" else mixed
    replay_frame, mining_report = _mine_v4c_frame(
        model=model_c,
        tokenizer=tokenizer_c,
        parent=parent_frame,
        batch_size=eval_batch_size,
        max_length=max_length,
        hard_negative_count=hard_negative_count,
        mining_weak_negative_rows=mining_weak_negative_rows,
        replay_total_rows=replay_total_rows,
    )

    # v4c: hard-negative continuation with mandatory ordinary replay.
    torch.cuda.reset_peak_memory_stats()
    stage_started = time.perf_counter()
    train_c = _train_model(
        model_c,
        tokenizer_c,
        replay_frame,
        batch_size=train_batch_size,
        epochs=v4c_epochs,
        learning_rate=v4c_lr,
        weight_decay=0.01,
        warmup_ratio=0.02,
        max_length=max_length,
        gradient_accumulation=gradient_accumulation,
        seed=SEED + 20,
    )
    eval_c, neural_c = _evaluate(
        model_c,
        tokenizer_c,
        valid,
        batch_size=eval_batch_size,
        max_length=max_length,
    )
    report_c, selected_c = _evaluate_stage(
        stage_name="v4c",
        valid=aligned,
        structured_scores=structured_scores,
        neural_scores=neural_c,
    )
    report_c["training"] = train_c
    report_c["neural_validation"] = eval_c
    report_c["parent_stage"] = parent_stage
    report_c["hard_negative_mining"] = mining_report
    report_c["seconds"] = float(time.perf_counter() - stage_started)
    report_c["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    report_c["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    dir_c = output_dir / "v4c-model"
    _save_checkpoint(model_c, tokenizer_c, dir_c, base_model_revision)
    stage_dirs["v4c"] = dir_c
    stage_reports["v4c"] = report_c
    stage_selected_scores["v4c"] = selected_c
    stage_neural_scores["v4c"] = neural_c
    stages_runtime["v4c"] = train_c

    improvement = select_v4_candidate(stage_reports)
    experimental_best = max(
        stage_reports,
        key=lambda name: float(stage_reports[name]["selected_macro_average_precision"]),
    )
    selected_stage = improvement[0] if improvement is not None else experimental_best
    selected_macro = float(stage_reports[selected_stage]["selected_macro_average_precision"])

    selected_dir = output_dir / "model"
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    shutil.copytree(stage_dirs[selected_stage], selected_dir)

    validation = aligned.copy()
    for name in ("v4a", "v4b", "v4c"):
        validation[f"neural_{name}"] = stage_neural_scores[name]
        validation[f"selected_{name}"] = stage_selected_scores[name]
    validation["selected_neural"] = stage_neural_scores[selected_stage]
    validation["selected_score"] = stage_selected_scores[selected_stage]
    validation.to_parquet(output_dir / "validation_predictions.parquet", index=False)

    total_seconds = time.perf_counter() - started
    payload = build_v4_metrics_payload(
        stages=stage_reports,
        selected_stage=selected_stage,
        selected_macro_average_precision=selected_macro,
        validation_rows=len(valid),
        validation_item_overlap=0,
        base_model=DEFAULT_MODEL,
        base_model_revision=base_model_revision,
        cuda_device=cuda_device,
        train_rows_human=len(human),
        train_rows_weak=int((mixed["source"].astype(str) == "weak").sum()),
        total_seconds=total_seconds,
        extra={
            "model_path_used": str(model_name),
            "max_length": int(max_length),
            "train_batch_size": int(train_batch_size),
            "eval_batch_size": int(eval_batch_size),
            "gradient_accumulation": int(gradient_accumulation),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "data_report": data_report,
            "structured_anchor": structured_metrics,
            "v4c_parent_stage": parent_stage,
            "experimental_best_stage": experimental_best,
            "strict_improvement_stage": improvement[0] if improvement is not None else None,
        },
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "version": "v4-strong-reranker",
        "base_model": DEFAULT_MODEL,
        "base_model_revision": base_model_revision.lower(),
        "selected_model_stage": selected_stage,
        "validation_macro_ap": selected_macro,
        "accepted_as_improvement": bool(payload["accepted_as_improvement"]),
        "max_length": int(max_length),
        "max_attrs": int(max_attrs),
        "max_chars": int(max_chars),
        "structured_column": STRUCTURED_COLUMN,
        "global_alpha_neural": float(stage_reports[selected_stage]["global_alpha_neural"]),
        "selected_blend": stage_reports[selected_stage]["selected_blend"],
        "category_alphas": stage_reports[selected_stage].get("shrunk_category_alphas", {}),
        "seed": SEED,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-items", required=True, type=Path)
    parser.add_argument("--human-matches", required=True, type=Path)
    parser.add_argument("--llm-matches", type=Path)
    parser.add_argument("--full-items", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--base-model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--v4a-epochs", type=float, default=1.0)
    parser.add_argument("--v4b-epochs", type=float, default=0.35)
    parser.add_argument("--v4c-epochs", type=float, default=1.0)
    parser.add_argument("--weak-presample-rows", type=int, default=1_000_000)
    parser.add_argument("--weak-final-rows", type=int, default=600_000)
    parser.add_argument("--hard-negative-count", type=int, default=20_000)
    parser.add_argument("--mining-weak-negative-rows", type=int, default=120_000)
    parser.add_argument("--replay-total-rows", type=int, default=80_000)
    parser.add_argument("--max-attrs", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=700)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()

    if args.benchmark_only:
        benchmark_v4(
            human_items_path=args.human_items,
            human_matches_path=args.human_matches,
            output_dir=args.output_dir,
            model_name=args.base_model,
            base_model_revision=args.base_model_revision,
            max_length=args.max_length,
            train_batch_size=args.train_batch_size,
            gradient_accumulation=args.gradient_accumulation,
            gradient_checkpointing=args.gradient_checkpointing,
            max_attrs=args.max_attrs,
            max_chars=args.max_chars,
        )
        return 0

    if args.llm_matches is None or args.full_items is None:
        parser.error("--llm-matches and --full-items are required for full v4 training")
    train_v4(
        human_items_path=args.human_items,
        human_matches_path=args.human_matches,
        llm_matches_path=args.llm_matches,
        full_items_path=args.full_items,
        output_dir=args.output_dir,
        model_name=args.base_model,
        base_model_revision=args.base_model_revision,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        gradient_checkpointing=args.gradient_checkpointing,
        v4a_epochs=args.v4a_epochs,
        v4b_epochs=args.v4b_epochs,
        v4c_epochs=args.v4c_epochs,
        weak_presample_rows=args.weak_presample_rows,
        weak_final_rows=args.weak_final_rows,
        hard_negative_count=args.hard_negative_count,
        mining_weak_negative_rows=args.mining_weak_negative_rows,
        replay_total_rows=args.replay_total_rows,
        max_attrs=args.max_attrs,
        max_chars=args.max_chars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
