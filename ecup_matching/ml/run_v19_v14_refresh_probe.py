"""Fast final-candidate ladder: refresh the exact v14 production checkpoint on unseen weak endpoints.

This is deliberately a refinement of the proven single-RuBERT runtime, not a new
runtime architecture.  The historical weak slice is reconstructed with the
same seed/config used by the v12/v14 family; candidate refresh rows touching any
of those endpoints are removed.  The remaining pool is split by connected
components so refresh-train and weak-evaluation items are disjoint.

Human development rows are used only as an in-sample *stability* diagnostic
because the starting v14 production model was fitted on all development rows.
They are never described as OOF validation.  Promotion therefore requires a
material gain on the item-disjoint weak population plus Brier/category/human
stability guards.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .run_v5_pretrained_biencoder import development_rows_and_folds
from .run_v7_outer_oof import IMMUTABLE_SPLIT_SHA, _stream_text_cache
from .run_v7_outer_oof_fast import _load_model_no_checkpoint
from .run_v7_outer_oof_frozen import _load_immutable_manifest
from .v5_evaluation import macro_ap_report
from .v7_runtime import predict_pairs
from .v17_weak_holdout import split_weak_item_disjoint
from .v18_ema import ExponentialMovingAverage
from .v18_neural import train_pair_phase_v18
from .run_v18_probe import _prepare_candidate_weak
from .v19_v14_refresh import filter_refresh_pairs, select_refresh_keeper


CANDIDATES: tuple[dict[str, object], ...] = (
    {
        "name": "v18-ema-r008",
        "epochs": 0.08,
        "learning_rate": 2.0e-6,
        "ema_decay": 0.99,
    },
    {
        "name": "v19-refresh-r005",
        "epochs": 0.05,
        "learning_rate": 2.0e-6,
        "ema_decay": None,
    },
    {
        "name": "v19-refresh-r010",
        "epochs": 0.10,
        "learning_rate": 1.5e-6,
        "ema_decay": None,
    },
)


def _sample_balanced(frame: pd.DataFrame, max_rows: int, seed: int, *, target_col: str) -> pd.DataFrame:
    if len(frame) <= int(max_rows):
        return frame.copy().reset_index(drop=True)
    work = frame.reset_index(drop=True).copy()
    hard = (pd.to_numeric(work[target_col], errors="raise").astype(float) >= 0.5).astype(np.int8)
    work["_sample_hard"] = hard
    groups = list(work.groupby(["category", "_sample_hard"], sort=True, dropna=False))
    quota = max(1, int(max_rows) // max(1, len(groups)))
    chosen: list[pd.DataFrame] = []
    used: set[int] = set()
    for number, (_, group) in enumerate(groups):
        take = min(quota, len(group))
        if take:
            part = group.sample(n=take, random_state=int(seed) + number)
            chosen.append(part)
            used.update(part.index.tolist())
    out = pd.concat(chosen, axis=0) if chosen else work.iloc[:0]
    need = int(max_rows) - len(out)
    if need > 0:
        pool = work.loc[~work.index.isin(used)]
        if len(pool):
            out = pd.concat(
                [out, pool.sample(n=min(need, len(pool)), random_state=int(seed) + 991)],
                axis=0,
            )
    return out.drop(columns="_sample_hard").sort_index(kind="mergesort").head(int(max_rows)).reset_index(drop=True)


def _score_weak(model, tokenizer, frame: pd.DataFrame, texts: dict[object, str], *, device: str, max_length: int) -> tuple[np.ndarray, dict[str, object]]:
    score, inference = predict_pairs(
        model=model,
        tokenizer=tokenizer,
        frame=frame,
        texts=texts,
        device=device,
        max_length=max_length,
        batch_size=64,
    )
    # split_weak_item_disjoint hardens held['target'] while retaining the
    # original probability in held['soft_target']; AP is therefore binary and
    # Brier remains calibrated against the original soft teacher probability.
    report = macro_ap_report(frame, score, target_col="target")
    soft = pd.to_numeric(frame["soft_target"], errors="raise").to_numpy(float)
    prob = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    return np.asarray(score, dtype=np.float64), {
        "rows": int(len(frame)),
        "macro_average_precision": float(report["macro_average_precision"]),
        "per_category_ap": report["per_category_ap"],
        "soft_brier": float(np.mean((prob - soft) ** 2)),
        "inference": inference,
    }


def _score_human(model, tokenizer, frame: pd.DataFrame, texts: dict[object, str], *, device: str, max_length: int) -> tuple[np.ndarray, dict[str, object]]:
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
    return np.asarray(score, dtype=np.float64), {
        "rows": int(len(frame)),
        "macro_average_precision": float(report["macro_average_precision"]),
        "per_category_ap": report["per_category_ap"],
        "inference": inference,
    }


def _gate_payload(human: dict[str, object], weak: dict[str, object], category_rows: dict[str, int]) -> dict[str, object]:
    return {
        "human_macro_average_precision": float(human["macro_average_precision"]),
        "weak_macro_average_precision": float(weak["macro_average_precision"]),
        "weak_soft_brier": float(weak["soft_brier"]),
        "per_category_ap": human["per_category_ap"],
        "category_row_counts": category_rows,
        "gold_metric_opened": False,
        "cross_split_item_overlap": 0,
    }


def run_probe(
    *,
    human_items_path: Path,
    human_matches_path: Path,
    weak_matches_path: Path,
    full_items_path: Path,
    base_model_dir: Path,
    output_dir: Path,
    max_length: int = 256,
    max_chars: int = 900,
    historical_presample_rows: int = 1_200_000,
    historical_final_rows: int = 600_000,
    candidate_presample_rows: int = 1_800_000,
    candidate_final_rows: int = 400_000,
    weak_eval_rows: int = 12_000,
    human_stability_rows: int = 10_000,
    seed: int = 2026,
) -> dict[str, object]:
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("fast v14 refresh requires CUDA")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    started = time.perf_counter()

    human_items = pd.read_parquet(human_items_path, columns=["id", "name", "attributes", "category"])
    human_matches = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    pairs, manifest, overlap = _load_immutable_manifest(
        human_items, human_matches, expected_split_sha=IMMUTABLE_SPLIT_SHA
    )
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(human_matches))
    dev = pairs.iloc[dev_rows].reset_index(drop=True)
    fold0 = dev.loc[fold_ids == 0, ["id1", "id2", "target", "category"]].reset_index(drop=True)
    human_stability = _sample_balanced(fold0, int(human_stability_rows), seed + 7001, target_col="target")
    category_rows = {
        str(key): int(value)
        for key, value in human_stability["category"].astype(str).value_counts().to_dict().items()
    }
    human_ids = set(human_stability["id1"].tolist()) | set(human_stability["id2"].tolist())
    human_subset = human_items[human_items["id"].isin(human_ids)].copy().reset_index(drop=True)
    if set(human_subset["id"].tolist()) != human_ids:
        raise RuntimeError("human stability item subset incomplete")
    human_texts = _stream_text_cache(human_subset, max_chars=max_chars)

    human_universe = set(human_matches["id1"].tolist()) | set(human_matches["id2"].tolist())
    historical, _, historical_report = _prepare_candidate_weak(
        weak_matches_path=weak_matches_path,
        full_items_path=full_items_path,
        forbidden_human_item_ids=human_universe,
        weak_presample_rows=int(historical_presample_rows),
        weak_final_rows=int(historical_final_rows),
        max_chars=max_chars,
        seed=int(seed),
        quality=False,
    )
    fresh, fresh_texts, fresh_report = _prepare_candidate_weak(
        weak_matches_path=weak_matches_path,
        full_items_path=full_items_path,
        forbidden_human_item_ids=human_universe,
        weak_presample_rows=int(candidate_presample_rows),
        weak_final_rows=int(candidate_final_rows),
        max_chars=max_chars,
        seed=int(seed) + 4242,
        quality=True,
    )
    fresh, endpoint_report = filter_refresh_pairs(fresh, historical)
    del historical
    gc.collect()
    if len(fresh) < 20_000:
        raise RuntimeError(f"too few endpoint-unseen weak pairs after filtering: {len(fresh)}")
    fresh = fresh.copy()
    fresh["soft_target"] = pd.to_numeric(fresh["target"], errors="raise").astype(float)
    refresh_train, weak_held, holdout = split_weak_item_disjoint(
        fresh, holdout_fraction=0.10, seed=int(seed) + 8123
    )
    del fresh
    gc.collect()
    weak_eval = _sample_balanced(weak_held, int(weak_eval_rows), seed + 9001, target_col="target")
    eval_ids = set(weak_eval["id1"].tolist()) | set(weak_eval["id2"].tolist())
    refresh_ids = set(refresh_train["id1"].tolist()) | set(refresh_train["id2"].tolist())
    if eval_ids & refresh_ids:
        raise RuntimeError("refresh/evaluation endpoint overlap")

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), local_files_only=True)
    baseline_model = _load_model_no_checkpoint(str(base_model_dir), last_n_layers=8, device=device)
    baseline_weak_score, baseline_weak = _score_weak(
        baseline_model, tokenizer, weak_eval, fresh_texts, device=device, max_length=max_length
    )
    baseline_human_score, baseline_human = _score_human(
        baseline_model, tokenizer, human_stability, human_texts, device=device, max_length=max_length
    )
    baseline_gate = _gate_payload(baseline_human, baseline_weak, category_rows)
    del baseline_model, baseline_weak_score, baseline_human_score
    gc.collect(); torch.cuda.empty_cache()

    candidate_gate_payloads: dict[str, dict[str, object]] = {}
    candidate_reports: dict[str, dict[str, object]] = {}
    model_dirs: dict[str, str] = {}
    for config in CANDIDATES:
        name = str(config["name"])
        model = _load_model_no_checkpoint(str(base_model_dir), last_n_layers=8, device=device)
        ema = None
        if config["ema_decay"] is not None:
            ema = ExponentialMovingAverage(model, decay=float(config["ema_decay"]))
        training = train_pair_phase_v18(
            model=model,
            tokenizer=tokenizer,
            frame=refresh_train,
            texts=fresh_texts,
            device=device,
            phase=name,
            epochs=float(config["epochs"]),
            physical_batch_size=32,
            effective_batch_size=32,
            max_length=max_length,
            learning_rate=float(config["learning_rate"]),
            ranking_weight=0.25,
            seed=int(seed) + 100,
            weak=True,
            ema=ema,
        )
        if ema is not None:
            ema.copy_to(model)
        weak_score, weak_report = _score_weak(
            model, tokenizer, weak_eval, fresh_texts, device=device, max_length=max_length
        )
        human_score, human_report = _score_human(
            model, tokenizer, human_stability, human_texts, device=device, max_length=max_length
        )
        gate_payload = _gate_payload(human_report, weak_report, category_rows)
        candidate_gate_payloads[name] = gate_payload
        candidate_reports[name] = {
            "config": config,
            "training": training.__dict__,
            "weak": weak_report,
            "human_stability": human_report,
        }
        candidate_dir = output_dir / "models" / name
        shutil.rmtree(candidate_dir, ignore_errors=True)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(candidate_dir, safe_serialization=True)
        tokenizer.save_pretrained(candidate_dir)
        model_dirs[name] = str(candidate_dir)
        del model, ema, weak_score, human_score
        gc.collect(); torch.cuda.empty_cache()

    selection = select_refresh_keeper(baseline_gate, candidate_gate_payloads)
    # Keep separate best EMA and best non-EMA candidates so the user gets two
    # independently useful submission variants when both families pass.
    evaluations = selection["evaluations"]
    passing_ema = [name for name in candidate_gate_payloads if name.startswith("v18-") and evaluations[name]["promote"]]
    passing_v19 = [name for name in candidate_gate_payloads if name.startswith("v19-") and evaluations[name]["promote"]]
    def best(names: list[str]) -> str | None:
        if not names:
            return None
        return max(names, key=lambda name: (float(evaluations[name]["weak_delta"]), float(evaluations[name]["human_delta"]), name))
    keepers = {"v18": best(passing_ema), "v19": best(passing_v19)}

    payload: dict[str, object] = {
        "version": "v19-v14-fast-refresh-probe-v1",
        "base_checkpoint": str(base_model_dir),
        "split_sha256": IMMUTABLE_SPLIT_SHA,
        "development_rows": int(len(dev_rows)),
        "sealed_gold_rows": int(len(manifest["gold_rows"])),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "cross_split_item_overlap": int(overlap["cross_split_item_overlap"]),
        "historical_weak": historical_report,
        "fresh_weak": fresh_report,
        "endpoint_filter": endpoint_report,
        "holdout": holdout,
        "refresh_train_rows": int(len(refresh_train)),
        "weak_eval_rows": int(len(weak_eval)),
        "human_stability_rows": int(len(human_stability)),
        "human_stability_is_not_oof": True,
        "baseline": {"weak": baseline_weak, "human_stability": baseline_human},
        "candidates": candidate_reports,
        "selection": selection,
        "keepers": keepers,
        "model_dirs": model_dirs,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "fast-refresh-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("FAST_REFRESH=" + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-items", type=Path, required=True)
    parser.add_argument("--human-matches", type=Path, required=True)
    parser.add_argument("--llm-matches", type=Path, required=True)
    parser.add_argument("--full-items", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-presample-rows", type=int, default=1_200_000)
    parser.add_argument("--historical-final-rows", type=int, default=600_000)
    parser.add_argument("--candidate-presample-rows", type=int, default=1_800_000)
    parser.add_argument("--candidate-final-rows", type=int, default=400_000)
    parser.add_argument("--weak-eval-rows", type=int, default=12_000)
    parser.add_argument("--human-stability-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run_probe(
        human_items_path=args.human_items,
        human_matches_path=args.human_matches,
        weak_matches_path=args.llm_matches,
        full_items_path=args.full_items,
        base_model_dir=args.base_model_dir,
        output_dir=args.output_dir,
        historical_presample_rows=args.historical_presample_rows,
        historical_final_rows=args.historical_final_rows,
        candidate_presample_rows=args.candidate_presample_rows,
        candidate_final_rows=args.candidate_final_rows,
        weak_eval_rows=args.weak_eval_rows,
        human_stability_rows=args.human_stability_rows,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
