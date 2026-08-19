"""Evaluate conservative weight-space rescues around the validated v14 refresh models.

The probe reuses the exact model checkpoints produced by the completed fast-final
run. It reconstructs the same deterministic item-disjoint weak evaluation slice
and human stability slice before scoring any interpolation. No new training is
performed here: only safetensors weight interpolation followed by the existing
v19 promotion gate.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from .run_v5_pretrained_biencoder import development_rows_and_folds
from .run_v7_outer_oof import IMMUTABLE_SPLIT_SHA, _stream_text_cache
from .run_v7_outer_oof_fast import _load_model_no_checkpoint
from .run_v7_outer_oof_frozen import _load_immutable_manifest
from .run_v18_probe import _prepare_candidate_weak
from .run_v19_v14_refresh_probe import _gate_payload, _sample_balanced, _score_human, _score_weak
from .v17_weak_holdout import split_weak_item_disjoint
from .v19_v14_refresh import filter_refresh_pairs, select_refresh_keeper
from .v19_weight_interp import interpolate_safetensors


_EXPECTED_BASELINE = {
    "weak_macro_average_precision": 0.7519230194571878,
    "weak_soft_brier": 0.1615403796697777,
    "human_macro_average_precision": 0.903981261891056,
}


def interpolation_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for alpha in (0.35, 0.40, 0.45, 0.50, 0.55):
        specs.append(
            {
                "name": f"v18-ema-rescue-a{int(round(alpha * 100)):03d}",
                "left": "base",
                "right": "v18",
                "alpha": float(alpha),
                "family": "v18",
            }
        )
    for alpha in (0.05, 0.10):
        specs.append(
            {
                "name": f"v19-sharpen-a{int(round(alpha * 100)):03d}",
                "left": "r005",
                "right": "r010",
                "alpha": float(alpha),
                "family": "v19",
            }
        )
    return specs


def _one_weight(model_dir: Path) -> Path:
    weights = sorted(model_dir.glob("*.safetensors"))
    if len(weights) != 1:
        raise RuntimeError(f"expected exactly one safetensors file in {model_dir}, got {weights}")
    return weights[0]


def _build_interpolated_model(
    *,
    left_dir: Path,
    right_dir: Path,
    output_dir: Path,
    alpha: float,
) -> None:
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.copytree(left_dir, output_dir)
    left_weight = _one_weight(left_dir)
    right_weight = _one_weight(right_dir)
    output_weight = output_dir / left_weight.name
    temp_weight = output_dir / (left_weight.name + ".tmp")
    interpolate_safetensors(left_weight, right_weight, temp_weight, alpha=float(alpha))
    temp_weight.replace(output_weight)


def _assert_baseline_reproduced(*, weak: dict[str, object], human: dict[str, object]) -> None:
    observed = {
        "weak_macro_average_precision": float(weak["macro_average_precision"]),
        "weak_soft_brier": float(weak["soft_brier"]),
        "human_macro_average_precision": float(human["macro_average_precision"]),
    }
    for key, expected in _EXPECTED_BASELINE.items():
        delta = abs(observed[key] - float(expected))
        if delta > 1e-6:
            raise RuntimeError(
                f"reconstructed validation drifted for {key}: observed={observed[key]} "
                f"expected={expected} abs_delta={delta}"
            )


def run_probe(
    *,
    human_items_path: Path,
    human_matches_path: Path,
    weak_matches_path: Path,
    full_items_path: Path,
    previous_root: Path,
    output_dir: Path,
    max_length: int = 256,
    max_chars: int = 900,
    seed: int = 2026,
) -> dict[str, object]:
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("weight interpolation probe requires CUDA")
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = {
        "base": previous_root / "v14-submission" / "model_v7_teacher",
        "v18": previous_root / "probe" / "models" / "v18-ema-r008",
        "r005": previous_root / "probe" / "models" / "v19-refresh-r005",
        "r010": previous_root / "probe" / "models" / "v19-refresh-r010",
    }
    for name, path in model_dirs.items():
        if not path.is_dir():
            raise FileNotFoundError(f"missing preserved {name} model: {path}")
        _one_weight(path)

    human_items = pd.read_parquet(
        human_items_path, columns=["id", "name", "attributes", "category"]
    )
    human_matches = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    pairs, manifest, overlap = _load_immutable_manifest(
        human_items, human_matches, expected_split_sha=IMMUTABLE_SPLIT_SHA
    )
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(human_matches))
    dev = pairs.iloc[dev_rows].reset_index(drop=True)
    fold0 = dev.loc[
        fold_ids == 0, ["id1", "id2", "target", "category"]
    ].reset_index(drop=True)
    human_stability = _sample_balanced(fold0, 10_000, seed + 7001, target_col="target")
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
        weak_presample_rows=1_200_000,
        weak_final_rows=600_000,
        max_chars=max_chars,
        seed=seed,
        quality=False,
    )
    fresh, fresh_texts, fresh_report = _prepare_candidate_weak(
        weak_matches_path=weak_matches_path,
        full_items_path=full_items_path,
        forbidden_human_item_ids=human_universe,
        weak_presample_rows=1_800_000,
        weak_final_rows=400_000,
        max_chars=max_chars,
        seed=seed + 4242,
        quality=True,
    )
    fresh, endpoint_report = filter_refresh_pairs(fresh, historical)
    del historical
    gc.collect()
    fresh = fresh.copy()
    fresh["soft_target"] = pd.to_numeric(fresh["target"], errors="raise").astype(float)
    refresh_train, weak_held, holdout = split_weak_item_disjoint(
        fresh, holdout_fraction=0.10, seed=seed + 8123
    )
    del fresh
    gc.collect()
    weak_eval = _sample_balanced(weak_held, 12_000, seed + 9001, target_col="target")
    eval_ids = set(weak_eval["id1"].tolist()) | set(weak_eval["id2"].tolist())
    train_ids = set(refresh_train["id1"].tolist()) | set(refresh_train["id2"].tolist())
    if eval_ids & train_ids:
        raise RuntimeError("refresh/evaluation endpoint overlap")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dirs["base"]), local_files_only=True)
    baseline_model = _load_model_no_checkpoint(
        str(model_dirs["base"]), last_n_layers=8, device="cuda"
    )
    _, baseline_weak = _score_weak(
        baseline_model, tokenizer, weak_eval, fresh_texts, device="cuda", max_length=max_length
    )
    _, baseline_human = _score_human(
        baseline_model,
        tokenizer,
        human_stability,
        human_texts,
        device="cuda",
        max_length=max_length,
    )
    _assert_baseline_reproduced(weak=baseline_weak, human=baseline_human)
    baseline_gate = _gate_payload(baseline_human, baseline_weak, category_rows)
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    reports: dict[str, dict[str, object]] = {}
    gates: dict[str, dict[str, object]] = {}
    generated_models: dict[str, str] = {}
    models_root = output_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    for spec in interpolation_specs():
        name = str(spec["name"])
        candidate_dir = models_root / name
        _build_interpolated_model(
            left_dir=model_dirs[str(spec["left"])],
            right_dir=model_dirs[str(spec["right"])],
            output_dir=candidate_dir,
            alpha=float(spec["alpha"]),
        )
        model = _load_model_no_checkpoint(str(candidate_dir), last_n_layers=8, device="cuda")
        _, weak_report = _score_weak(
            model, tokenizer, weak_eval, fresh_texts, device="cuda", max_length=max_length
        )
        _, human_report = _score_human(
            model,
            tokenizer,
            human_stability,
            human_texts,
            device="cuda",
            max_length=max_length,
        )
        gates[name] = _gate_payload(human_report, weak_report, category_rows)
        reports[name] = {
            "spec": spec,
            "weak": weak_report,
            "human_stability": human_report,
        }
        generated_models[name] = str(candidate_dir)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    selection = select_refresh_keeper(baseline_gate, gates)
    evaluations = selection["evaluations"]

    def best_for_family(family: str) -> str | None:
        names = [
            str(spec["name"])
            for spec in interpolation_specs()
            if spec["family"] == family and evaluations[str(spec["name"])]["promote"]
        ]
        if not names:
            return None
        return max(
            names,
            key=lambda name: (
                float(evaluations[name]["weak_delta"]),
                float(evaluations[name]["human_delta"]),
                name,
            ),
        )

    keepers = {"v18": best_for_family("v18"), "v19": best_for_family("v19")}
    payload: dict[str, object] = {
        "version": "v19-weight-interp-probe-v1",
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
        "candidates": reports,
        "selection": selection,
        "keepers": keepers,
        "model_dirs": generated_models,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "weight-interp-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("WEIGHT_INTERP=" + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-items", type=Path, required=True)
    parser.add_argument("--human-matches", type=Path, required=True)
    parser.add_argument("--llm-matches", type=Path, required=True)
    parser.add_argument("--full-items", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_probe(
        human_items_path=args.human_items,
        human_matches_path=args.human_matches,
        weak_matches_path=args.llm_matches,
        full_items_path=args.full_items,
        previous_root=args.previous_root,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
