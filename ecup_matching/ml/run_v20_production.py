"""D10 full-development v20 refit using the exact selected causal mode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import pandas as pd

from .run_v5_pretrained_biencoder import development_rows_and_folds
from .run_v7_outer_oof import IMMUTABLE_SPLIT_SHA, _stream_text_cache
from .run_v7_outer_oof_fast import _load_model_no_checkpoint
from .run_v7_outer_oof_frozen import _load_immutable_manifest
from .run_v18_probe import _prepare_candidate_weak
from .run_v20_probe import candidate_flags, _mixed_frame, _source_frame, _train_plain, _train_rationale
from .textnorm import normalize_item
from .train_v4_reranker import DEFAULT_MODEL_REVISION, _verify_model_revision
from .v20_corpus import build_gold_corpus, balanced_sample
from .v20_neural import V20MultiTaskModel, production_base_model
from .v20_policy import V20Policy, policy_sha256
from .v20_strata import classify_pair_stratum
from .v7_runtime import build_v7_text_cache_from_parquet


def run_v20_production(
    *, candidate: str, human_items_path: Path, human_matches_path: Path,
    weak_matches_path: Path, generated_labels_path: Path, full_items_path: Path,
    output_dir: Path, model_path: str, base_model_revision: str,
    weak_presample_rows: int = 3_000_000, weak_final_rows: int = 1_500_000,
    weak_epochs: float = 1.0, max_length: int = 256, max_chars: int = 900,
    physical_batch_size: int = 32, learning_rate: float = 1.5e-5,
    weak_learning_rate: float = 1e-5, ranking_weight: float = 0.25,
    seed: int = 2026, apply_v19_refresh: bool = False,
) -> dict[str, object]:
    import torch
    from transformers import AutoTokenizer

    flags = candidate_flags(candidate)
    if candidate == "control":
        raise ValueError("control cannot be packaged as v20 production")
    if not torch.cuda.is_available():
        raise RuntimeError("v20 production requires CUDA")
    _verify_model_revision(model_path, base_model_revision)
    policy = V20Policy()
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    human_items = pd.read_parquet(human_items_path, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    pairs, manifest, overlap = _load_immutable_manifest(human_items, matches, expected_split_sha=IMMUTABLE_SPLIT_SHA)
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(matches))
    if len(dev_rows) != 285_210 or len(fold_ids) != 285_210 or len(manifest.get("gold_rows", [])) != 80_444:
        raise RuntimeError("immutable split row counts changed")
    human_train = pairs.iloc[dev_rows][["id1", "id2", "target", "category"]].reset_index(drop=True)
    human_universe = set(matches.id1) | set(matches.id2)
    norm_map = {
        r.id: normalize_item(r.id, r.name, r.attributes, r.category)
        for r in human_items.loc[human_items.id.isin(set(human_train.id1) | set(human_train.id2))].itertuples(index=False)
    }
    human_train["reason_code"] = [
        classify_pair_stratum(norm_map[r.id1], norm_map[r.id2]).reason_code
        for r in human_train.itertuples(index=False)
    ]
    human_train = _source_frame(human_train, "human")

    weak, _, weak_report = _prepare_candidate_weak(
        weak_matches_path=weak_matches_path, full_items_path=full_items_path,
        forbidden_human_item_ids=human_universe, weak_presample_rows=weak_presample_rows,
        weak_final_rows=weak_final_rows, max_chars=max_chars, seed=seed, quality=True,
    )
    generated = pd.read_parquet(generated_labels_path).reset_index(drop=True)
    if (set(generated.id1) | set(generated.id2)) & human_universe:
        raise RuntimeError("production generated labels touch human universe")
    weak_source = _source_frame(weak, "historical_weak")
    generated_source = _source_frame(generated, "generated_llm")
    generated_source["weak_weight"] = 1.0
    other_gold, other_report = build_gold_corpus(pd.DataFrame(), weak_source, generated_source, forbidden_ids=set(), seed=seed)
    human_gold, human_report = build_gold_corpus(human_train, pd.DataFrame(), pd.DataFrame(), forbidden_ids=set(), seed=seed + 3)
    phase_a_rows = min(max(1, int(round(len(weak) * weak_epochs))), len(other_gold))
    other_phase = balanced_sample(other_gold, phase_a_rows, seed=seed + 10)
    other_phase["weak_weight"] = other_phase["match_weight"].astype(float)

    nonhuman_ids = set(other_gold.id1) | set(other_gold.id2)
    nonhuman_texts = build_v7_text_cache_from_parquet(full_items_path, nonhuman_ids, max_chars=max_chars)
    human_texts = _stream_text_cache(human_items.loc[human_items.id.isin(set(human_gold.id1) | set(human_gold.id2))], max_chars=max_chars)
    texts = {**nonhuman_texts, **human_texts}

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    base_model = _load_model_no_checkpoint(model_path, last_n_layers=8, device="cuda")
    model = V20MultiTaskModel(base_model, reason_classes=11) if flags["rationale"] else base_model
    train = _train_rationale if flags["rationale"] else _train_plain
    training = []
    kwargs = dict(model=model, tokenizer=tokenizer, frame=other_phase, texts=texts,
                  phase="v20-production-A", epochs=1.0, batch=physical_batch_size,
                  max_length=max_length, lr=weak_learning_rate, ranking_weight=ranking_weight, seed=seed)
    if flags["rationale"]: kwargs["policy"] = policy
    else: kwargs["weak"] = True
    training.append(train(**kwargs).__dict__)

    if flags["replay"]:
        phases = [
            ("B", policy.phase_b_human_to_other, max(len(human_gold), 640), 1.0, seed + 101),
            ("C", policy.phase_c_human_to_other, len(human_gold), policy.phase_c_lr_multiplier, seed + 202),
        ]
        for name, ratio, total, lr_mult, s in phases:
            mixed = _mixed_frame(human_gold, other_gold, ratio=ratio, total_rows=total, seed=s)
            kwargs = dict(model=model, tokenizer=tokenizer, frame=mixed, texts=texts,
                          phase=f"v20-production-{name}", epochs=1.0, batch=physical_batch_size,
                          max_length=max_length, lr=learning_rate * lr_mult,
                          ranking_weight=ranking_weight, seed=s)
            if flags["rationale"]: kwargs["policy"] = policy
            else: kwargs["weak"] = True
            training.append(train(**kwargs).__dict__)
    else:
        kwargs = dict(model=model, tokenizer=tokenizer, frame=human_gold, texts=texts,
                      phase="v20-production-human", epochs=1.0, batch=physical_batch_size,
                      max_length=max_length, lr=learning_rate, ranking_weight=ranking_weight, seed=seed + 100)
        if flags["rationale"]: kwargs["policy"] = policy
        else: kwargs["weak"] = False
        training.append(train(**kwargs).__dict__)

    if apply_v19_refresh:
        training.append(_train_plain(
            model=production_base_model(model), tokenizer=tokenizer, frame=weak_source,
            texts=texts, phase="v20-production-v19-refresh", epochs=0.05,
            batch=physical_batch_size, max_length=max_length, lr=2e-6,
            ranking_weight=ranking_weight, seed=seed + 1900, weak=True,
        ).__dict__)

    production = production_base_model(model)
    model_dir = output_dir / "model_v7_teacher"
    model_dir.mkdir(parents=True, exist_ok=True)
    production.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    weights = sorted(p.name for p in model_dir.glob("*.safetensors"))
    if len(weights) != 1:
        raise RuntimeError(f"expected exactly one production checkpoint, got {weights}")
    payload = {
        "version": "v20-production-refit", "candidate": candidate,
        "candidate_flags": flags, "is_production_refit": True,
        "validation_metric_reported": False, "base_model": "ai-forever/ruBert-base",
        "base_model_revision": base_model_revision.lower(), "max_length": max_length,
        "inference_batch_size": 64, "split_sha256": IMMUTABLE_SPLIT_SHA,
        "development_rows": int(len(dev_rows)), "training_rows": int(len(human_train)),
        "sealed_gold_rows": int(len(manifest["gold_rows"])), "gold_metric_opened": False,
        "gold_rows_scored": 0, "cross_split_item_overlap": int(overlap["cross_split_item_overlap"]),
        "weak_presample_rows": int(weak_presample_rows), "weak_final_rows": int(weak_final_rows),
        "weak_epochs": float(weak_epochs), "generated_rows": int(len(generated)),
        "v19_refresh_applied": bool(apply_v19_refresh), "policy_sha256": policy_sha256(policy),
        "weak_preparation": weak_report, "other_corpus": other_report,
        "human_corpus": human_report, "training_phases": training,
        "saved_files": sorted(p.name for p in model_dir.iterdir() if p.is_file()),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "production-metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print("V20_PRODUCTION=" + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", choices=[c for c in ("data-only", "rationale", "replay-data", "replay-rationale")], required=True)
    p.add_argument("--human-items", type=Path, required=True)
    p.add_argument("--human-matches", type=Path, required=True)
    p.add_argument("--llm-matches", type=Path, required=True)
    p.add_argument("--generated-labels", type=Path, required=True)
    p.add_argument("--full-items", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--base-model", default="/opt/models/rubert-base")
    p.add_argument("--base-model-revision", default=DEFAULT_MODEL_REVISION)
    p.add_argument("--weak-presample-rows", type=int, default=3_000_000)
    p.add_argument("--weak-final-rows", type=int, default=1_500_000)
    p.add_argument("--weak-epochs", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-chars", type=int, default=900)
    p.add_argument("--physical-batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1.5e-5)
    p.add_argument("--weak-learning-rate", type=float, default=1e-5)
    p.add_argument("--ranking-weight", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--apply-v19-refresh", action="store_true")
    a = p.parse_args()
    run_v20_production(
        candidate=a.candidate, human_items_path=a.human_items, human_matches_path=a.human_matches,
        weak_matches_path=a.llm_matches, generated_labels_path=a.generated_labels,
        full_items_path=a.full_items, output_dir=a.output_dir, model_path=a.base_model,
        base_model_revision=a.base_model_revision, weak_presample_rows=a.weak_presample_rows,
        weak_final_rows=a.weak_final_rows, weak_epochs=a.weak_epochs, max_length=a.max_length,
        max_chars=a.max_chars, physical_batch_size=a.physical_batch_size,
        learning_rate=a.learning_rate, weak_learning_rate=a.weak_learning_rate,
        ranking_weight=a.ranking_weight, seed=a.seed, apply_v19_refresh=a.apply_v19_refresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
