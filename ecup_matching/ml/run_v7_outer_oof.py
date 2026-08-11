from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .features_v2 import build_features_v2_chunked
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .run_v5_validation_audit import build_split_descriptors
from .train_v1 import attach_pair_category
from .train_v2_structured import prefilter_weak_candidates_parquet
from .train_v4_reranker import DEFAULT_MODEL_REVISION, _verify_model_revision
from .v5_evaluation import macro_ap_report
from .v5_validation import build_v5_split_manifest, manifest_sha256, validate_manifest_no_overlap
from .v7_item_text import serialize_item_v7
from .v7_neural import (
    build_v7_text_cache_from_parquet,
    configure_trainable_layers,
    predict_pairs,
    train_pair_phase,
)
from .v7_teacher_contract import V7TeacherConfig, validate_v7_teacher_config
from .weak_labels import prepare_weak_pairs, sample_weak_training
from .textnorm import normalize_item


IMMUTABLE_SPLIT_SHA = "aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b"


def _phase(name: str, **payload) -> None:
    print(json.dumps({"phase": name, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def _stream_text_cache(items: pd.DataFrame, *, max_chars: int) -> dict[object, str]:
    required = {"id", "name", "attributes", "category"}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"items missing v7 text columns: {sorted(missing)}")
    out: dict[object, str] = {}
    total = len(items)
    started = time.perf_counter()
    for index, (item_id, name, attributes, category) in enumerate(
        items[["id", "name", "attributes", "category"]].itertuples(index=False, name=None),
        start=1,
    ):
        norm = normalize_item(item_id, name, attributes, category)
        out[item_id] = f"[CAT] {norm.category}\n{serialize_item_v7(norm, max_chars=max_chars)}"
        if index == 1 or index % 100_000 == 0 or index == total:
            elapsed = time.perf_counter() - started
            _phase(
                "serialize-items",
                done=index,
                total=total,
                percent=round(100.0 * index / max(total, 1), 2),
                elapsed_seconds=round(elapsed, 2),
                items_per_second=round(index / max(elapsed, 1e-9), 2),
            )
    return out


def _load_model(model_path: str, *, last_n_layers: int, device: str):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    configure_trainable_layers(model, last_n_layers=last_n_layers)
    return model.to(device)


def _build_immutable_manifest(
    human_items: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    expected_split_sha: str,
) -> tuple[pd.DataFrame, dict, dict[str, int]]:
    pairs = attach_pair_category(matches.copy(), human_items).reset_index(drop=True)
    _phase("split-features", rows=len(pairs))
    features = build_features_v2_chunked(
        human_items,
        pairs,
        attribute_importance=None,
        chunk_size=25_000,
    )
    descriptors = build_split_descriptors(pairs, features)
    manifest = build_v5_split_manifest(
        pairs,
        descriptors,
        gold_fraction=0.22,
        n_folds=5,
        seed=2026,
    )
    actual = manifest_sha256(manifest)
    if actual != expected_split_sha:
        raise ValueError(f"immutable split SHA mismatch: expected={expected_split_sha}, actual={actual}")
    overlap = validate_manifest_no_overlap(pairs, manifest)
    if int(overlap["cross_split_item_overlap"]) != 0:
        raise RuntimeError("immutable split contains item overlap")
    return pairs, manifest, overlap


def _prepare_common_weak(
    *,
    weak_matches_path: Path,
    full_items_path: Path,
    forbidden_human_item_ids: set[object],
    weak_presample_rows: int,
    weak_final_rows: int,
    max_chars: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[object, str], dict[str, object]]:
    _phase(
        "weak-prefilter",
        forbidden_human_items=len(forbidden_human_item_ids),
        max_presample_rows=weak_presample_rows,
    )
    weak, weak_input_rows = prefilter_weak_candidates_parquet(
        weak_matches_path,
        validation_item_ids=forbidden_human_item_ids,
        max_presample_rows=weak_presample_rows,
        seed=seed,
    )
    weak, prep = prepare_weak_pairs(weak[["id1", "id2", "target"]])
    endpoint_overlap = (set(weak["id1"].tolist()) | set(weak["id2"].tolist())) & forbidden_human_item_ids
    if endpoint_overlap:
        raise RuntimeError(f"common weak corpus contains {len(endpoint_overlap)} human items")

    intermediate_cap = min(len(weak), max(weak_final_rows, int(round(weak_final_rows * 1.5))))
    weak = sample_weak_training(weak, max_rows=intermediate_cap, seed=seed)

    # First pass is intentionally category-only: it is needed before the final
    # category-balanced weak sampling and avoids materializing huge attribute blobs.
    weak_ids = set(weak["id1"].tolist()) | set(weak["id2"].tolist())
    category_items = select_items_by_ids(full_items_path, weak_ids, include_attributes=False)
    weak = attach_pair_category(weak, category_items)
    weak = sample_weak_training(weak, max_rows=weak_final_rows, seed=seed + 17)

    final_ids = set(weak["id1"].tolist()) | set(weak["id2"].tolist())
    if final_ids & forbidden_human_item_ids:
        raise RuntimeError("forbidden human item survived common weak selection")

    # Second pass is the quality-critical one: stream real attributes only for the
    # final selected IDs and immediately serialize them, rather than substituting {}.
    weak_texts, category_by_id = build_v7_text_cache_from_parquet(
        full_items_path,
        final_ids,
        max_chars=max_chars,
    )
    left_category = weak["id1"].map(category_by_id)
    right_category = weak["id2"].map(category_by_id)
    if left_category.isna().any() or right_category.isna().any():
        raise RuntimeError("full-attribute weak scan missed selected item IDs")
    if (left_category.astype(str) != right_category.astype(str)).any():
        raise RuntimeError("full-attribute weak scan found cross-category pair")
    if (weak["category"].astype(str) != left_category.astype(str)).any():
        raise RuntimeError("weak category changed between category-only and full-attribute scans")

    report: dict[str, object] = {
        "input_rows": int(weak_input_rows),
        "prepared": prep,
        "selected_rows": int(len(weak)),
        "selected_items": int(len(weak_texts)),
        "selected_items_with_real_attributes": int(len(weak_texts)),
        "forbidden_human_items": int(len(forbidden_human_item_ids)),
        "human_item_overlap": 0,
    }
    del category_items, category_by_id
    return (
        weak[["id1", "id2", "target", "category", "weak_weight"]].reset_index(drop=True),
        weak_texts,
        report,
    )


def run_v7_outer_oof(
    *,
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
    human_epochs: float = 2.0,
    physical_batch_size: int = 2,
    effective_batch_size: int = 32,
    trainable_last_layers: int = 8,
    learning_rate: float = 1.5e-5,
    weak_learning_rate: float = 1.0e-5,
    ranking_weight: float = 0.25,
    seed: int = 2026,
) -> dict[str, object]:
    import torch
    from transformers import AutoTokenizer

    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("v7 outer OOF requires CUDA")
    if expected_split_sha != IMMUTABLE_SPLIT_SHA:
        raise ValueError("v7 training may only use the immutable split SHA")
    validate_v7_teacher_config(
        V7TeacherConfig(
            max_length=max_length,
            curriculum_rows=max(1, weak_final_rows),
            effective_batch_size=effective_batch_size,
            epochs=max(weak_epochs, 1e-6),
            max_steps=None,
        )
    )
    _verify_model_revision(model_path, base_model_revision)
    device = "cuda"
    cuda_device = torch.cuda.get_device_name(0)
    _phase("start", cuda_device=cuda_device, max_length=max_length, max_chars=max_chars)

    human_items = pd.read_parquet(
        human_items_path, columns=["id", "name", "attributes", "category"]
    )
    matches = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    pairs, manifest, overlap = _build_immutable_manifest(
        human_items,
        matches,
        expected_split_sha=expected_split_sha,
    )
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(matches))
    if len(dev_rows) != 285_210:
        raise RuntimeError(f"unexpected development row count: {len(dev_rows)}")
    if len(manifest.get("gold_rows", [])) != 80_444:
        raise RuntimeError("unexpected sealed gold row count")
    dev = pairs.iloc[dev_rows].reset_index(drop=True)
    if len(np.unique(fold_ids)) != 5:
        raise RuntimeError("expected exactly five outer folds")

    human_item_universe = set(matches["id1"].tolist()) | set(matches["id2"].tolist())
    weak, weak_texts, weak_report = _prepare_common_weak(
        weak_matches_path=weak_matches_path,
        full_items_path=full_items_path,
        forbidden_human_item_ids=human_item_universe,
        weak_presample_rows=weak_presample_rows,
        weak_final_rows=weak_final_rows,
        max_chars=max_chars,
        seed=seed,
    )
    weak_needed = set(weak["id1"].tolist()) | set(weak["id2"].tolist())
    if not weak_needed <= set(weak_texts):
        raise RuntimeError("common weak text cache is incomplete")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    common_model = _load_model(model_path, last_n_layers=trainable_last_layers, device=device)
    weak_training = train_pair_phase(
        model=common_model,
        tokenizer=tokenizer,
        frame=weak,
        texts=weak_texts,
        device=device,
        phase="common-weak-pretrain",
        epochs=weak_epochs,
        physical_batch_size=physical_batch_size,
        effective_batch_size=effective_batch_size,
        max_length=max_length,
        learning_rate=weak_learning_rate,
        ranking_weight=ranking_weight,
        seed=seed,
        weak=True,
        telemetry_every_steps=100,
    )

    checkpoint_fd, checkpoint_name = tempfile.mkstemp(prefix="ecup-v7-common-", suffix=".pt")
    os.close(checkpoint_fd)
    checkpoint_path = Path(checkpoint_name)
    torch.save(common_model.state_dict(), checkpoint_path)
    del common_model, weak_texts, weak
    gc.collect()
    torch.cuda.empty_cache()

    fold_metrics: list[dict[str, object]] = []
    oof_rows: list[np.ndarray] = []
    oof_scores: list[np.ndarray] = []
    fold_training_seconds = 0.0
    inference_seconds = 0.0
    try:
        for held_fold in range(5):
            fold_started = time.perf_counter()
            train_mask = fold_ids != held_fold
            held_mask = fold_ids == held_fold
            human_train = dev.loc[train_mask, ["id1", "id2", "target", "category"]].reset_index(drop=True)
            held = dev.loc[held_mask, ["id1", "id2", "target", "category"]].reset_index(drop=True)
            held_rows = dev_rows[held_mask]
            _phase(
                "fold-start",
                fold=held_fold,
                train_rows=len(human_train),
                held_rows=len(held),
            )

            needed = (
                set(human_train["id1"].tolist())
                | set(human_train["id2"].tolist())
                | set(held["id1"].tolist())
                | set(held["id2"].tolist())
            )
            fold_items = human_items[human_items["id"].isin(needed)].copy().reset_index(drop=True)
            if set(fold_items["id"].tolist()) != needed:
                raise RuntimeError(f"fold {held_fold} item subset is incomplete")
            fold_texts = _stream_text_cache(fold_items, max_chars=max_chars)

            model = _load_model(model_path, last_n_layers=trainable_last_layers, device="cpu")
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state, strict=True)
            del state
            model = model.to(device)
            configure_trainable_layers(model, last_n_layers=trainable_last_layers)
            training = train_pair_phase(
                model=model,
                tokenizer=tokenizer,
                frame=human_train,
                texts=fold_texts,
                device=device,
                phase=f"human-fold-{held_fold}",
                epochs=human_epochs,
                physical_batch_size=physical_batch_size,
                effective_batch_size=effective_batch_size,
                max_length=max_length,
                learning_rate=learning_rate,
                ranking_weight=ranking_weight,
                seed=seed + held_fold * 101,
                weak=False,
                telemetry_every_steps=100,
            )
            score, inference = predict_pairs(
                model=model,
                tokenizer=tokenizer,
                frame=held,
                texts=fold_texts,
                device=device,
                max_length=max_length,
                batch_size=16,
            )
            report = macro_ap_report(held, score)
            fold_elapsed = time.perf_counter() - fold_started
            fold_training_seconds += training.elapsed_seconds
            inference_seconds += float(inference["seconds"])
            fold_payload: dict[str, object] = {
                "fold": held_fold,
                "train_rows": int(len(human_train)),
                "held_rows": int(len(held)),
                "macro_average_precision": float(report["macro_average_precision"]),
                "per_category_ap": report["per_category_ap"],
                "training": training.__dict__,
                "inference": inference,
                "elapsed_seconds": float(fold_elapsed),
            }
            fold_metrics.append(fold_payload)
            oof_rows.append(np.asarray(held_rows, dtype=np.int64))
            oof_scores.append(np.asarray(score, dtype=np.float64))
            pd.DataFrame(
                {
                    "row_index": held_rows,
                    "fold": held_fold,
                    "v7_teacher_score": score,
                }
            ).sort_values("row_index").to_parquet(
                output_dir / f"v7-fold-{held_fold}-oof.parquet", index=False
            )
            (output_dir / f"v7-fold-{held_fold}-metrics.json").write_text(
                json.dumps(fold_payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _phase(
                "fold-complete",
                fold=held_fold,
                macro_average_precision=report["macro_average_precision"],
                elapsed_seconds=round(fold_elapsed, 2),
            )
            del model, fold_texts, fold_items
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        checkpoint_path.unlink(missing_ok=True)

    row_index = np.concatenate(oof_rows)
    scores = np.concatenate(oof_scores)
    order = np.argsort(row_index, kind="stable")
    row_index = row_index[order]
    scores = scores[order]
    if row_index.tolist() != dev_rows.tolist():
        raise RuntimeError("v7 OOF row identity mismatch")
    strict_report = macro_ap_report(pairs.iloc[row_index].reset_index(drop=True), scores)
    fold_scores = [float(item["macro_average_precision"]) for item in fold_metrics]
    elapsed_seconds = time.perf_counter() - started
    payload: dict[str, object] = {
        "version": "v7-outer-oof",
        "candidate": "identity-first-256-macro-balanced-shared-weak-rubert-base-full-attrs",
        "base_model": "ai-forever/ruBert-base",
        "base_model_revision": base_model_revision.lower(),
        "cuda_device": cuda_device,
        "max_length": int(max_length),
        "max_chars": int(max_chars),
        "trainable_last_layers": int(trainable_last_layers),
        "weak_epochs": float(weak_epochs),
        "human_epochs": float(human_epochs),
        "physical_batch_size": int(physical_batch_size),
        "effective_batch_size": int(effective_batch_size),
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev_rows)),
        "sealed_gold_rows": int(len(manifest["gold_rows"])),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "cross_split_item_overlap": int(overlap["cross_split_item_overlap"]),
        "outer_folds_complete": int(len(fold_metrics)),
        "strict_oof_macro_average_precision": float(strict_report["macro_average_precision"]),
        "per_category_ap": strict_report["per_category_ap"],
        "fold_macro_ap": fold_scores,
        "folds": fold_metrics,
        "common_weak": weak_report,
        "common_weak_training": weak_training.__dict__,
        "fold_training_seconds": float(fold_training_seconds),
        "inference_seconds": float(inference_seconds),
        "elapsed_seconds": float(elapsed_seconds),
    }
    pd.DataFrame(
        {
            "row_index": row_index,
            "fold": fold_ids,
            "v7_teacher_score": scores,
        }
    ).to_parquet(output_dir / "v7-development-oof.parquet", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    _phase(
        "complete",
        strict_oof_macro_average_precision=payload["strict_oof_macro_average_precision"],
        elapsed_seconds=round(elapsed_seconds, 2),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--human-epochs", type=float, default=2.0)
    parser.add_argument("--physical-batch-size", type=int, default=2)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--trainable-last-layers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--weak-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    payload = run_v7_outer_oof(
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
        trainable_last_layers=args.trainable_last_layers,
        learning_rate=args.learning_rate,
        weak_learning_rate=args.weak_learning_rate,
        ranking_weight=args.ranking_weight,
        seed=args.seed,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())