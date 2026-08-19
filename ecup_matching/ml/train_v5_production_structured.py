from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .data_subset import select_items_by_ids
from .features import normalize_items
from .features_v2 import build_pair_features_v2
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .train_v1 import attach_pair_category
from .train_v2_structured import candidate_sample_weights, prefilter_weak_candidates_parquet
from .v5_category_specialists import fit_category_specialists
from .v5_explicit_attributes import (
    build_explicit_attribute_features,
    build_explicit_leaf_cache,
    learn_explicit_attribute_keys,
)
from .v5_validation import manifest_sha256
from .weak_labels import prepare_weak_pairs, remove_human_conflicts, sample_weak_training


def _legacy_modules(legacy_package_dir: Path):
    package_dir = legacy_package_dir.resolve()
    if package_dir.name != "legacy_ecup":
        raise ValueError("legacy package directory must be named legacy_ecup")
    sys.path.insert(0, str(package_dir.parent))
    return (
        importlib.import_module("legacy_ecup.ml.features"),
        importlib.import_module("legacy_ecup.ml.features_v2"),
        importlib.import_module("legacy_ecup.ml.v5_sparse"),
    )


def _fit_explicit_bundle(
    *,
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    item_cache,
    base_builder,
    canonical_values: bool,
    seed: int,
) -> dict:
    leaf_cache = build_explicit_leaf_cache(item_cache, canonical_values=canonical_values)
    spec = learn_explicit_attribute_keys(
        items,
        pairs,
        max_keys_per_category=40,
        min_support=30,
        item_cache=item_cache,
        leaf_cache=leaf_cache,
    )
    models: dict[str, HistGradientBoostingClassifier] = {}
    for category in sorted(pairs["category"].astype(str).unique().tolist()):
        mask = pairs["category"].astype(str) == category
        category_pairs = pairs.loc[mask].reset_index(drop=True)
        base = base_builder(items, category_pairs, item_cache=item_cache)
        explicit = build_explicit_attribute_features(
            items,
            category_pairs,
            spec,
            item_cache=item_cache,
            category=category,
            leaf_cache=leaf_cache,
        )
        x = pd.concat(
            [base.drop(columns=["category"]).reset_index(drop=True), explicit.reset_index(drop=True)],
            axis=1,
        ).to_numpy(dtype=np.float32)
        y = category_pairs["target"].to_numpy(dtype=np.int8)
        if len(np.unique(y)) < 2:
            raise ValueError(f"category {category!r} does not contain both target classes")
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.06,
            max_iter=350,
            max_leaf_nodes=31,
            min_samples_leaf=15,
            l2_regularization=3.0,
            early_stopping=False,
            random_state=int(seed),
        )
        model.fit(x, y)
        models[category] = model
    return {
        "models": models,
        "key_spec": spec,
        "canonical_values": bool(canonical_values),
    }


def train_structured_bundle(
    *,
    human_items_path: Path,
    full_items_path: Path,
    matches_path: Path,
    weak_matches_path: Path,
    manifest_path: Path,
    base_oof_path: Path,
    legacy_package_dir: Path,
    output_path: Path,
    expected_split_sha: str,
    seed: int = 2026,
) -> dict:
    started = time.perf_counter()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = manifest_sha256(manifest)
    if actual_sha != expected_split_sha:
        raise ValueError(f"sealed split SHA mismatch: {actual_sha}")

    legacy_features, legacy_features_v2, legacy_sparse = _legacy_modules(legacy_package_dir)
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, _ = development_rows_and_folds(manifest, total_rows=len(matches))
    dev = matches.iloc[dev_rows].reset_index(drop=True)
    base_oof = pd.read_parquet(base_oof_path, columns=["row_index", "score"]).sort_values("row_index")
    if base_oof["row_index"].astype(np.int64).tolist() != dev_rows.tolist():
        raise ValueError("development base OOF rows do not align with sealed split")

    dev_ids = pd.unique(pd.concat([dev["id1"], dev["id2"]], ignore_index=True))
    human_items = select_items_by_ids(human_items_path, dev_ids, include_attributes=True)
    dev = attach_pair_category(dev, human_items)
    if dev["category"].isna().any():
        raise RuntimeError("failed to attach development category")

    legacy_base = legacy_features_v2.build_features_v2_chunked(
        human_items, dev, attribute_importance=None, chunk_size=25_000
    )

    gold_rows = np.asarray(manifest.get("gold_rows", []), dtype=np.int64)
    if len(gold_rows):
        if gold_rows.min() < 0 or gold_rows.max() >= len(matches):
            raise IndexError("manifest contains out-of-range gold row")
        gold_pairs = matches.iloc[gold_rows]
        forbidden = set(gold_pairs["id1"].tolist()) | set(gold_pairs["id2"].tolist())
    else:
        forbidden = set()

    weak, weak_input_rows = prefilter_weak_candidates_parquet(
        weak_matches_path,
        validation_item_ids=forbidden,
        max_presample_rows=250_000,
        seed=seed,
    )
    weak, prepare_report = prepare_weak_pairs(weak[["id1", "id2", "target"]])
    weak, conflict_report = remove_human_conflicts(weak, dev[["id1", "id2", "target"]])
    weak_ids = set(weak["id1"].tolist()) | set(weak["id2"].tolist())
    weak_items = select_items_by_ids(full_items_path, weak_ids, include_attributes=True)
    weak = attach_pair_category(weak, weak_items)
    weak = sample_weak_training(weak, max_rows=150_000, seed=seed)
    final_weak_ids = set(weak["id1"].tolist()) | set(weak["id2"].tolist())
    if final_weak_ids & forbidden:
        raise RuntimeError("weak production curriculum contains sealed-gold item")
    weak_items = weak_items[weak_items["id"].isin(final_weak_ids)].reset_index(drop=True)
    legacy_weak = legacy_features_v2.build_features_v2_chunked(
        weak_items, weak, attribute_importance=None, chunk_size=25_000
    )
    weak_x = pd.concat([legacy_base, legacy_weak], ignore_index=True)
    weak_y = np.concatenate(
        [dev["target"].to_numpy(dtype=np.int8), weak["hard_target"].to_numpy(dtype=np.int8)]
    )
    weak_categories = pd.concat(
        [dev["category"].reset_index(drop=True), weak["category"].reset_index(drop=True)],
        ignore_index=True,
    )
    weak_sources = pd.Series(["human"] * len(dev) + ["weak"] * len(weak))
    weak_confidence = np.concatenate(
        [np.ones(len(dev), dtype=np.float64), weak["weak_weight"].to_numpy(dtype=np.float64)]
    )
    weak_weights = candidate_sample_weights(
        weak_categories,
        weak_sources,
        weak_y,
        weak_confidence,
        weak_x["hard_negative_score"].to_numpy(dtype=np.float64),
        hard_negative_boost=0.0,
    )
    weak_bundle = fit_category_specialists(
        weak_x,
        weak_y,
        sample_weight=weak_weights,
        seed=seed,
        max_iter=300,
        min_samples_leaf=15,
        l2_regularization=2.0,
    )

    sparse_encoder = legacy_sparse.fit_sparse_item_encoder(
        human_items,
        max_char_features=120_000,
        max_word_features=60_000,
    )
    sparse_features = legacy_sparse.transform_sparse_pairs(sparse_encoder, human_items, dev)
    sparse_x = pd.concat(
        [legacy_base.reset_index(drop=True), sparse_features.reset_index(drop=True)], axis=1
    )
    sparse_bundle = fit_category_specialists(
        sparse_x,
        dev["target"].to_numpy(dtype=np.int8),
        seed=seed,
        max_iter=300,
        min_samples_leaf=15,
        l2_regularization=2.0,
    )

    legacy_cache = legacy_features.normalize_items(human_items)
    explicit_bundle = _fit_explicit_bundle(
        items=human_items,
        pairs=dev,
        item_cache=legacy_cache,
        base_builder=legacy_features_v2.build_pair_features_v2,
        canonical_values=False,
        seed=seed,
    )

    typed_cache = normalize_items(human_items)
    typed_explicit_bundle = _fit_explicit_bundle(
        items=human_items,
        pairs=dev,
        item_cache=typed_cache,
        base_builder=build_pair_features_v2,
        canonical_values=True,
        seed=seed,
    )

    bundle = {
        "version": "v5-best-structured-production-v1",
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev)),
        "gold_rows_used": 0,
        "legacy_source_commit": "cb350b4e7ba6",
        "weak": weak_bundle,
        "sparse": {"encoder": sparse_encoder, "specialists": sparse_bundle},
        "explicit": explicit_bundle,
        "typed_explicit": typed_explicit_bundle,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path, compress=3)
    return {
        "version": bundle["version"],
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev)),
        "gold_rows_used": 0,
        "weak_input_rows": int(weak_input_rows),
        "weak_final_rows": int(len(weak)),
        "weak_prepare": prepare_report,
        "weak_conflicts": conflict_report,
        "sparse_items": int(len(human_items)),
        "legacy_explicit_keys": int(sum(len(v) for v in explicit_bundle["key_spec"].values())),
        "typed_explicit_keys": int(sum(len(v) for v in typed_explicit_bundle["key_spec"].values())),
        "output_bytes": int(output_path.stat().st_size),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--human-items", type=Path, required=True)
    p.add_argument("--full-items", type=Path, required=True)
    p.add_argument("--matches", type=Path, required=True)
    p.add_argument("--weak-matches", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--base-oof", type=Path, required=True)
    p.add_argument("--legacy-package-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-split-sha", required=True)
    args = p.parse_args()
    result = train_structured_bundle(
        human_items_path=args.human_items,
        full_items_path=args.full_items,
        matches_path=args.matches,
        weak_matches_path=args.weak_matches,
        manifest_path=args.manifest,
        base_oof_path=args.base_oof,
        legacy_package_dir=args.legacy_package_dir,
        output_path=args.output,
        expected_split_sha=args.expected_split_sha,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
