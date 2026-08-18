"""Fold-0 probe that varies weak exposure and reports two axes instead of one.

Every candidate from v12 through v15 trained with `--weak-final-rows 600000
--weak-epochs 0.35`, which is `210,000` weak examples seen out of a pool of
`11,187,780` — `1.88%`. The setting was never varied. Meanwhile the only
change that ever moved the public leaderboard materially was weak supervision:
v7 -> v12 gained `+0.0142` externally on a `+0.0035` local move, while every
architecture change since (residual, graph, late interaction, MoE, setwise)
moved it by `0.002` or less, or hurt.

So weak exposure is the one lever with demonstrated external effect and it has
never been pushed. This driver pushes it.

It also fixes the measurement problem that made previous selections unreliable.
Human fold-0 ranked v13b first of four candidates; the leaderboard ranked it
third. Human labelling covers `5.31%` of the item universe and the weak pool
covers `92.44%` with zero shared ids, so fold-0 is a narrow view of a large
population. This driver additionally holds out an item-disjoint slice of the
weak corpus and reports Macro AP on it, before and after the human phase.

The weak axis is scored against weak targets, so it measures agreement with
the LLM annotator on unseen items, not human truth. Both numbers are reported;
neither is presented as a leaderboard estimate.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .run_v5_pretrained_biencoder import development_rows_and_folds
from .run_v7_outer_oof import (
    IMMUTABLE_SPLIT_SHA,
    _build_immutable_manifest,
    _phase,
    _prepare_common_weak,
    _stream_text_cache,
)
from .run_v7_outer_oof_fast import _load_model_no_checkpoint
from .train_v4_reranker import DEFAULT_MODEL_REVISION, _verify_model_revision
from .v5_evaluation import macro_ap_report
from .v7_neural import predict_pairs, train_pair_phase
from .v17_weak_holdout import split_weak_item_disjoint


V13B_FOLD0_REFERENCE = 0.7086611385531062
V13B_WEAK_EXAMPLES_SEEN = 210_000


def _weak_macro_ap(
    *,
    model,
    tokenizer,
    frame: pd.DataFrame,
    texts,
    max_length: int,
    stage: str,
) -> dict[str, object]:
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
    payload = {
        "stage": stage,
        "rows": int(len(frame)),
        "macro_average_precision": float(report["macro_average_precision"]),
        "per_category_ap": report["per_category_ap"],
        "inference": inference,
    }
    _phase(f"weak-holdout-{stage}", **{k: v for k, v in payload.items() if k != "per_category_ap"})
    return payload


def run_weakscale_probe(
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
    weak_holdout_fraction: float = 0.05,
    human_epochs: float = 1.00,
    physical_batch_size: int = 32,
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
        raise RuntimeError("v17 weak-scale probe requires CUDA")
    if expected_split_sha != IMMUTABLE_SPLIT_SHA:
        raise ValueError("probe may only use the immutable split SHA")
    _verify_model_revision(model_path, base_model_revision)
    cuda_device = torch.cuda.get_device_name(0)
    _phase(
        "v17-probe-start",
        fold=0,
        weak_final_rows=int(weak_final_rows),
        weak_epochs=float(weak_epochs),
        v13b_reference=V13B_FOLD0_REFERENCE,
        cuda_device=cuda_device,
    )

    human_items = pd.read_parquet(
        human_items_path, columns=["id", "name", "attributes", "category"]
    )
    matches = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    pairs, manifest, overlap = _build_immutable_manifest(
        human_items, matches, expected_split_sha=expected_split_sha
    )
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(matches))
    if len(dev_rows) != 285_210 or len(manifest.get("gold_rows", [])) != 80_444:
        raise RuntimeError("immutable split row counts changed")
    dev = pairs.iloc[dev_rows].reset_index(drop=True)
    held_fold = 0
    train_mask = fold_ids != held_fold
    held_mask = fold_ids == held_fold
    human_train = dev.loc[train_mask, ["id1", "id2", "target", "category"]].reset_index(drop=True)
    held = dev.loc[held_mask, ["id1", "id2", "target", "category"]].reset_index(drop=True)
    held_rows = dev_rows[held_mask]

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
    if (set(weak["id1"].tolist()) | set(weak["id2"].tolist())) & human_item_universe:
        raise RuntimeError("weak corpus leaked a human item")

    weak_train, weak_held, holdout_report = split_weak_item_disjoint(
        weak, holdout_fraction=weak_holdout_fraction, seed=seed + 977
    )
    del weak
    gc.collect()
    weak_examples_seen = int(round(len(weak_train) * float(weak_epochs)))
    _phase(
        "v17-weak-holdout",
        weak_examples_seen=weak_examples_seen,
        exposure_vs_v13b=weak_examples_seen / V13B_WEAK_EXAMPLES_SEEN,
        **holdout_report,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = _load_model_no_checkpoint(
        model_path, last_n_layers=trainable_last_layers, device="cuda"
    )
    weak_training = train_pair_phase(
        model=model,
        tokenizer=tokenizer,
        frame=weak_train,
        texts=weak_texts,
        device="cuda",
        phase="v17-common-weak",
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
    del weak_train
    gc.collect()
    torch.cuda.empty_cache()

    weak_after_weak = _weak_macro_ap(
        model=model,
        tokenizer=tokenizer,
        frame=weak_held,
        texts=weak_texts,
        max_length=max_length,
        stage="after-weak-phase",
    )

    needed = (
        set(human_train["id1"].tolist())
        | set(human_train["id2"].tolist())
        | set(held["id1"].tolist())
        | set(held["id2"].tolist())
    )
    fold_items = human_items[human_items["id"].isin(needed)].copy().reset_index(drop=True)
    if set(fold_items["id"].tolist()) != needed:
        raise RuntimeError("probe item subset is incomplete")
    fold_texts = _stream_text_cache(fold_items, max_chars=max_chars)
    human_training = train_pair_phase(
        model=model,
        tokenizer=tokenizer,
        frame=human_train,
        texts=fold_texts,
        device="cuda",
        phase="v17-human-fold0",
        epochs=human_epochs,
        physical_batch_size=physical_batch_size,
        effective_batch_size=effective_batch_size,
        max_length=max_length,
        learning_rate=learning_rate,
        ranking_weight=ranking_weight,
        seed=seed,
        weak=False,
        telemetry_every_steps=100,
    )

    score, inference = predict_pairs(
        model=model,
        tokenizer=tokenizer,
        frame=held,
        texts=fold_texts,
        device="cuda",
        max_length=max_length,
        batch_size=64,
    )
    report = macro_ap_report(held, score)
    fold0_ap = float(report["macro_average_precision"])

    weak_after_human = _weak_macro_ap(
        model=model,
        tokenizer=tokenizer,
        frame=weak_held,
        texts=weak_texts,
        max_length=max_length,
        stage="after-human-phase",
    )

    payload: dict[str, object] = {
        "version": "v17-weakscale-probe",
        "diagnostic_only": True,
        "fold": 0,
        "base_model": "ai-forever/ruBert-base",
        "base_model_revision": base_model_revision.lower(),
        "cuda_device": cuda_device,
        "max_length": int(max_length),
        "max_chars": int(max_chars),
        "weak_presample_rows": int(weak_presample_rows),
        "weak_final_rows": int(weak_final_rows),
        "weak_epochs": float(weak_epochs),
        "weak_examples_seen": weak_examples_seen,
        "exposure_vs_v13b": weak_examples_seen / V13B_WEAK_EXAMPLES_SEEN,
        "human_epochs": float(human_epochs),
        "physical_batch_size": int(physical_batch_size),
        "effective_batch_size": int(effective_batch_size),
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev_rows)),
        "sealed_gold_rows": int(len(manifest["gold_rows"])),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "cross_split_item_overlap": int(overlap["cross_split_item_overlap"]),
        "train_rows": int(len(human_train)),
        "held_rows": int(len(held)),
        "fold0_macro_average_precision": fold0_ap,
        "v13b_fold0_reference": V13B_FOLD0_REFERENCE,
        "delta_vs_v13b_fold0": float(fold0_ap - V13B_FOLD0_REFERENCE),
        "per_category_ap": report["per_category_ap"],
        "weak_holdout": holdout_report,
        "weak_holdout_after_weak_phase": weak_after_weak,
        "weak_holdout_after_human_phase": weak_after_human,
        "common_weak": weak_report,
        "weak_training": weak_training.__dict__,
        "human_training": human_training.__dict__,
        "inference": inference,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    pd.DataFrame(
        {
            "row_index": held_rows,
            "fold": held_fold,
            "v17_probe_score": np.asarray(score, dtype=np.float64),
        }
    ).to_parquet(output_dir / "v17-weakscale-fold0-oof.parquet", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--human-epochs", type=float, default=1.00)
    parser.add_argument("--physical-batch-size", type=int, default=32)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--trainable-last-layers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--weak-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    run_weakscale_probe(
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
        trainable_last_layers=args.trainable_last_layers,
        learning_rate=args.learning_rate,
        weak_learning_rate=args.weak_learning_rate,
        ranking_weight=args.ranking_weight,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
