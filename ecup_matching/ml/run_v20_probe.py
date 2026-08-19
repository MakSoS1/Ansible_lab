"""D5-D9 v20 probe with causally isolated data, rationale and replay mechanisms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .run_v5_pretrained_biencoder import development_rows_and_folds
from .run_v7_outer_oof import IMMUTABLE_SPLIT_SHA, _stream_text_cache
from .run_v7_outer_oof_fast import _load_model_no_checkpoint
from .run_v7_outer_oof_frozen import _load_immutable_manifest
from .run_v18_probe import _prepare_candidate_weak, _score_weak
from .train_v4_reranker import DEFAULT_MODEL_REVISION, _verify_model_revision
from .v5_evaluation import macro_ap_report
from .v7_runtime import build_v7_text_cache_from_parquet, predict_pairs
from .v17_weak_holdout import split_weak_item_disjoint
from .v18_neural import train_pair_phase_v18
from .v20_corpus import build_gold_corpus, balanced_sample
from .v20_neural import V20MultiTaskModel, production_base_model, train_v20_phase
from .v20_policy import V20Policy, policy_sha256, validate_fold_exclusion


CANDIDATES = ("control", "data-only", "rationale", "replay-data", "replay-rationale")
TAIL_CATEGORIES = (
    "Электроника", "Одежда", "Обувь", "Ювелирные изделия",
    "Галантерея и аксессуары", "Мебель",
)


def candidate_flags(candidate: str) -> dict[str, bool]:
    modes = {
        "control": {"generated": False, "rationale": False, "replay": False},
        "data-only": {"generated": True, "rationale": False, "replay": False},
        "rationale": {"generated": True, "rationale": True, "replay": False},
        "replay-data": {"generated": True, "rationale": False, "replay": True},
        "replay-rationale": {"generated": True, "rationale": True, "replay": True},
    }
    if candidate not in modes:
        raise ValueError(f"unsupported v20 candidate: {candidate}")
    return dict(modes[candidate])


def _source_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    out["source"] = source
    if "reason_code" not in out:
        out["reason_code"] = "OTHER"
    if "stratum_reliability" not in out:
        out["stratum_reliability"] = 1.0
    if "admitted" not in out:
        out["admitted"] = True
    if "weak_weight" not in out:
        out["weak_weight"] = 1.0
    return out


def _remove_endpoint_overlap(frame: pd.DataFrame, forbidden: set[object]) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame.copy(), 0
    mask = frame["id1"].isin(forbidden) | frame["id2"].isin(forbidden)
    return frame.loc[~mask].reset_index(drop=True), int(mask.sum())


def _sample_rows(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[:0].copy()
    return frame.sample(n=int(n), replace=len(frame) < int(n), random_state=int(seed)).reset_index(drop=True)


def _mixed_frame(human: pd.DataFrame, other: pd.DataFrame, *, ratio: tuple[int, int], total_rows: int, seed: int) -> pd.DataFrame:
    h, o = map(int, ratio)
    human_n = int(round(total_rows * h / (h + o)))
    left = _sample_rows(human, human_n, seed)
    right = _sample_rows(other, total_rows - human_n, seed + 1)
    out = pd.concat([left, right], ignore_index=True, sort=False)
    out["weak_weight"] = out["match_weight"].astype(float)
    return out.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def _tail_metric(per_category: dict[str, float]) -> float:
    values = [float(per_category[name]) for name in TAIL_CATEGORIES if name in per_category]
    return float(np.mean(values)) if values else float("nan")


def _score_proxy(model, tokenizer, proxy: pd.DataFrame, texts: dict, *, max_length: int) -> dict[str, object]:
    score, inference = predict_pairs(
        model=model, tokenizer=tokenizer, frame=proxy, texts=texts,
        device="cuda", max_length=max_length, batch_size=64,
    )
    report = macro_ap_report(proxy, score)
    return {
        "macro_average_precision": float(report["macro_average_precision"]),
        "per_category_ap": report["per_category_ap"],
        "tail_macro_average_precision": _tail_metric(report["per_category_ap"]),
        "inference": inference,
    }


def _train_plain(*, model, tokenizer, frame, texts, phase, epochs, batch, max_length, lr, ranking_weight, seed, weak):
    work = frame.copy()
    if "weak_weight" not in work and "match_weight" in work:
        work["weak_weight"] = work["match_weight"].astype(float)
    return train_pair_phase_v18(
        model=model, tokenizer=tokenizer, frame=work, texts=texts, device="cuda",
        phase=phase, epochs=epochs, physical_batch_size=batch,
        effective_batch_size=max(batch, 32), max_length=max_length,
        learning_rate=lr, ranking_weight=ranking_weight, seed=seed, weak=weak,
    )


def _train_rationale(*, model, tokenizer, frame, texts, phase, epochs, batch, max_length, lr, ranking_weight, seed, policy):
    return train_v20_phase(
        model=model, tokenizer=tokenizer, frame=frame, texts=texts, device="cuda",
        phase=phase, epochs=epochs, physical_batch_size=batch,
        effective_batch_size=max(batch, 32), max_length=max_length,
        learning_rate=lr, ranking_weight=ranking_weight, seed=seed,
        lambda_reason=policy.lambda_reason, lambda_consistency=policy.lambda_consistency,
    )


def run_v20_probe(
    *, candidate: str, fold: int, human_items_path: Path, human_matches_path: Path,
    human_training_path: Path, weak_matches_path: Path, generated_labels_path: Path,
    proxy_pairs_path: Path, full_items_path: Path, output_dir: Path, model_path: str,
    base_model_revision: str, weak_presample_rows: int, weak_final_rows: int,
    weak_epochs: float, max_length: int = 256, max_chars: int = 900,
    weak_holdout_fraction: float = 0.05, physical_batch_size: int = 32,
    learning_rate: float = 1.5e-5, weak_learning_rate: float = 1e-5,
    ranking_weight: float = 0.25, seed: int = 2026, apply_v19_refresh: bool = False,
) -> dict[str, object]:
    import torch
    from transformers import AutoTokenizer

    flags = candidate_flags(candidate)
    if not torch.cuda.is_available():
        raise RuntimeError("canonical v20 probe requires CUDA")
    if fold not in range(5):
        raise ValueError("fold must be 0..4")
    _verify_model_revision(model_path, base_model_revision)
    policy = V20Policy()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    human_items = pd.read_parquet(human_items_path, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    pairs, manifest, overlap = _load_immutable_manifest(human_items, matches, expected_split_sha=IMMUTABLE_SPLIT_SHA)
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(matches))
    if len(dev_rows) != 285_210 or len(manifest.get("gold_rows", [])) != 80_444:
        raise RuntimeError("immutable split row counts changed")
    dev = pairs.iloc[dev_rows].reset_index(drop=True)
    held_mask = fold_ids == fold
    held = dev.loc[held_mask, ["id1", "id2", "target", "category"]].reset_index(drop=True)
    held_rows = dev_rows[held_mask]
    held_items = set(held.id1) | set(held.id2)

    human_train = pd.read_parquet(human_training_path).reset_index(drop=True)
    validate_fold_exclusion(human_train, held_items)
    human_train = _source_frame(human_train, "human")
    human_universe = set(matches.id1) | set(matches.id2)

    historical, _, weak_preparation = _prepare_candidate_weak(
        weak_matches_path=weak_matches_path, full_items_path=full_items_path,
        forbidden_human_item_ids=human_universe, weak_presample_rows=weak_presample_rows,
        weak_final_rows=weak_final_rows, max_chars=max_chars, seed=seed, quality=True,
    )
    historical = historical.copy()
    historical["soft_target"] = pd.to_numeric(historical["target"], errors="raise").astype(float)
    weak_train, weak_held, weak_holdout = split_weak_item_disjoint(
        historical, holdout_fraction=weak_holdout_fraction, seed=seed + 977
    )
    weak_held_items = set(weak_held.id1) | set(weak_held.id2)

    generated = pd.read_parquet(generated_labels_path).reset_index(drop=True)
    generated, removed_weak = _remove_endpoint_overlap(generated, weak_held_items)
    proxy = pd.read_parquet(proxy_pairs_path).reset_index(drop=True)
    if proxy.empty or not {"id1", "id2", "target", "category"}.issubset(proxy.columns):
        raise ValueError("proxy must contain labelled pairs")
    proxy_items = set(proxy.id1) | set(proxy.id2)
    generated, removed_proxy = _remove_endpoint_overlap(generated, proxy_items)
    if (set(generated.id1) | set(generated.id2)) & human_universe:
        raise RuntimeError("generated training labels touch human item universe")
    if proxy_items & human_universe:
        raise RuntimeError("proxy touches human item universe")
    if proxy_items & (set(weak_train.id1) | set(weak_train.id2) | weak_held_items):
        raise RuntimeError("proxy touches selected historical weak item universe")

    weak_source = _source_frame(weak_train, "historical_weak")
    generated_source = _source_frame(generated, "generated_llm")
    generated_source["weak_weight"] = 1.0
    other_gold, other_report = build_gold_corpus(
        pd.DataFrame(), weak_source,
        generated_source if flags["generated"] else generated_source.iloc[:0],
        forbidden_ids=set(), seed=seed,
    )
    human_gold, human_report = build_gold_corpus(
        human_train, pd.DataFrame(), pd.DataFrame(), forbidden_ids=held_items, seed=seed + 3
    )
    base_examples = max(1, int(round(len(weak_train) * weak_epochs)))
    other_phase = balanced_sample(other_gold, min(base_examples, len(other_gold)), seed=seed + 10)
    other_phase["weak_weight"] = other_phase["match_weight"].astype(float)

    nonhuman_ids = set(other_gold.id1) | set(other_gold.id2) | weak_held_items | proxy_items
    nonhuman_texts = build_v7_text_cache_from_parquet(full_items_path, nonhuman_ids, max_chars=max_chars)
    human_ids = set(human_gold.id1) | set(human_gold.id2) | held_items
    fold_items = human_items.loc[human_items.id.isin(human_ids)].copy().reset_index(drop=True)
    if set(fold_items.id) != human_ids:
        raise RuntimeError("human text cache is incomplete")
    texts = {**nonhuman_texts, **_stream_text_cache(fold_items, max_chars=max_chars)}

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    base_model = _load_model_no_checkpoint(model_path, last_n_layers=8, device="cuda")
    model = V20MultiTaskModel(base_model, reason_classes=11) if flags["rationale"] else base_model
    training = []
    train = _train_rationale if flags["rationale"] else _train_plain
    kwargs = dict(model=model, tokenizer=tokenizer, frame=other_phase, texts=texts,
                  phase=f"v20-{candidate}-A", epochs=1.0, batch=physical_batch_size,
                  max_length=max_length, lr=weak_learning_rate, ranking_weight=ranking_weight,
                  seed=seed)
    if flags["rationale"]:
        kwargs["policy"] = policy
    else:
        kwargs["weak"] = True
    training.append(train(**kwargs).__dict__)

    if flags["replay"]:
        for suffix, ratio, total, lr_mult, phase_seed in [
            ("B", policy.phase_b_human_to_other, max(len(human_gold), 640), 1.0, seed + 101),
            ("C", policy.phase_c_human_to_other, len(human_gold), policy.phase_c_lr_multiplier, seed + 202),
        ]:
            mixed = _mixed_frame(human_gold, other_gold, ratio=ratio, total_rows=total, seed=phase_seed)
            kwargs = dict(model=model, tokenizer=tokenizer, frame=mixed, texts=texts,
                          phase=f"v20-{candidate}-{suffix}", epochs=1.0, batch=physical_batch_size,
                          max_length=max_length, lr=learning_rate * lr_mult, ranking_weight=ranking_weight,
                          seed=phase_seed)
            if flags["rationale"]:
                kwargs["policy"] = policy
            else:
                kwargs["weak"] = True
            training.append(train(**kwargs).__dict__)
    else:
        kwargs = dict(model=model, tokenizer=tokenizer, frame=human_gold, texts=texts,
                      phase=f"v20-{candidate}-human", epochs=1.0, batch=physical_batch_size,
                      max_length=max_length, lr=learning_rate, ranking_weight=ranking_weight,
                      seed=seed + 100)
        if flags["rationale"]:
            kwargs["policy"] = policy
        else:
            kwargs["weak"] = False
        training.append(train(**kwargs).__dict__)

    if apply_v19_refresh:
        training.append(_train_plain(
            model=production_base_model(model), tokenizer=tokenizer, frame=weak_train, texts=texts,
            phase="v20-v19-refresh", epochs=0.05, batch=physical_batch_size,
            max_length=max_length, lr=2e-6, ranking_weight=ranking_weight,
            seed=seed + 1900, weak=True,
        ).__dict__)

    scorer = production_base_model(model)
    human_score, human_inference = predict_pairs(
        model=scorer, tokenizer=tokenizer, frame=held, texts=texts,
        device="cuda", max_length=max_length, batch_size=64,
    )
    human_metrics = macro_ap_report(held, human_score)
    _, weak_metrics = _score_weak(
        scorer, tokenizer, weak_held, texts, device="cuda", max_length=max_length, stage="v20-after-training"
    )
    proxy_metrics = _score_proxy(scorer, tokenizer, proxy, texts, max_length=max_length)
    payload = {
        "version": "v20-probe-v2", "candidate": candidate, "candidate_flags": flags, "fold": fold,
        "policy_sha256": policy_sha256(policy), "split_sha256": IMMUTABLE_SPLIT_SHA,
        "development_rows": int(len(dev_rows)), "sealed_gold_rows": int(len(manifest["gold_rows"])),
        "gold_metric_opened": False, "gold_rows_scored": 0,
        "cross_split_item_overlap": int(overlap["cross_split_item_overlap"]),
        "human_train_rows": int(len(human_gold)), "held_rows": int(len(held)),
        "historical_weak_rows": int(len(weak_train)), "generated_rows": int(len(generated)),
        "generated_weak_overlap_removed": removed_weak, "generated_proxy_overlap_removed": removed_proxy,
        "weak_presample_rows": int(weak_presample_rows), "weak_final_rows": int(weak_final_rows),
        "weak_epochs": float(weak_epochs), "v19_refresh_applied": bool(apply_v19_refresh),
        "human_macro_average_precision": float(human_metrics["macro_average_precision"]),
        "human_per_category_ap": human_metrics["per_category_ap"],
        "human_tail_macro_average_precision": _tail_metric(human_metrics["per_category_ap"]),
        "weak_holdout": weak_holdout, "weak_metrics": weak_metrics,
        "proxy_metrics": proxy_metrics, "weak_preparation": weak_preparation,
        "other_corpus": other_report, "human_corpus": human_report,
        "training_phases": training, "human_inference": human_inference,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    pd.DataFrame({"row_index": held_rows, "fold": fold, "v20_score": np.asarray(human_score)}).to_parquet(
        output_dir / "v20-fold-oof.parquet", index=False
    )
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print("V20_PROBE=" + json.dumps({
        "candidate": candidate, "fold": fold,
        "human": payload["human_macro_average_precision"],
        "proxy": proxy_metrics["macro_average_precision"],
        "weak": weak_metrics["macro_average_precision"],
    }, ensure_ascii=False), flush=True)
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", choices=CANDIDATES, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--human-items", type=Path, required=True)
    p.add_argument("--human-matches", type=Path, required=True)
    p.add_argument("--human-training", type=Path, required=True)
    p.add_argument("--llm-matches", type=Path, required=True)
    p.add_argument("--generated-labels", type=Path, required=True)
    p.add_argument("--proxy-pairs", type=Path, required=True)
    p.add_argument("--full-items", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--base-model", default="/opt/models/rubert-base")
    p.add_argument("--base-model-revision", default=DEFAULT_MODEL_REVISION)
    p.add_argument("--weak-presample-rows", type=int, default=1_200_000)
    p.add_argument("--weak-final-rows", type=int, default=600_000)
    p.add_argument("--weak-epochs", type=float, default=0.35)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-chars", type=int, default=900)
    p.add_argument("--weak-holdout-fraction", type=float, default=0.05)
    p.add_argument("--physical-batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1.5e-5)
    p.add_argument("--weak-learning-rate", type=float, default=1e-5)
    p.add_argument("--ranking-weight", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--apply-v19-refresh", action="store_true")
    a = p.parse_args()
    run_v20_probe(
        candidate=a.candidate, fold=a.fold, human_items_path=a.human_items,
        human_matches_path=a.human_matches, human_training_path=a.human_training,
        weak_matches_path=a.llm_matches, generated_labels_path=a.generated_labels,
        proxy_pairs_path=a.proxy_pairs, full_items_path=a.full_items,
        output_dir=a.output_dir, model_path=a.base_model,
        base_model_revision=a.base_model_revision, weak_presample_rows=a.weak_presample_rows,
        weak_final_rows=a.weak_final_rows, weak_epochs=a.weak_epochs,
        max_length=a.max_length, max_chars=a.max_chars,
        weak_holdout_fraction=a.weak_holdout_fraction,
        physical_batch_size=a.physical_batch_size, learning_rate=a.learning_rate,
        weak_learning_rate=a.weak_learning_rate, ranking_weight=a.ranking_weight,
        seed=a.seed, apply_v19_refresh=a.apply_v19_refresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
