from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .category_attrs import learn_attribute_importance
from .data_subset import select_items_by_ids
from .features_v2 import FEATURE_NAMES_V2, build_features_v2_chunked
from .label_graph import augment_transitive_positives, clean_human_pairs
from .metrics import macro_average_precision
from .model_io import save_model_bundle
from .train_v1 import attach_pair_category, category_equalizing_weights
from .v2_split import fixed_v1_split
from .weak_labels import prepare_weak_pairs, remove_human_conflicts, sample_weak_training


SEED = 2026
V1_MACRO_AP = 0.49616548946964434


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def fit_estimator(
    features: pd.DataFrame,
    y,
    sample_weight=None,
    seed: int = SEED,
    max_iter: int = 350,
) -> Pipeline:
    if "category" not in features.columns:
        raise ValueError("features must contain category")
    numeric = [c for c in features.columns if c != "category"]
    preprocess = ColumnTransformer(
        [("category", _one_hot_encoder(), ["category"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.07,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=seed,
    )
    model = Pipeline([("preprocess", preprocess), ("classifier", classifier)])
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["classifier__sample_weight"] = np.asarray(sample_weight, dtype=np.float64)
    model.fit(features[["category", *numeric]], np.asarray(y, dtype=np.int8), **fit_kwargs)
    return model


def _weak_weight_vector(target: pd.Series) -> np.ndarray:
    p = pd.to_numeric(target, errors="raise").to_numpy(dtype=float)
    if ((p < 0) | (p > 1)).any():
        raise ValueError("weak target must be in [0,1]")
    weight = np.zeros(len(p), dtype=np.float32)
    extreme = (p <= 0.03) | (p >= 0.97)
    strong = ((p > 0.03) & (p <= 0.15)) | ((p >= 0.85) & (p < 0.97))
    medium = ((p > 0.15) & (p <= 0.30)) | ((p >= 0.70) & (p < 0.85))
    weight[extreme] = 1.0
    weight[strong] = 0.6
    weight[medium] = 0.3
    return weight


def prefilter_weak_candidates(
    weak: pd.DataFrame,
    validation_item_ids: set[object],
    max_presample_rows: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """Cheap vectorized gate before expensive weak-label dedupe/item loading."""
    if max_presample_rows <= 0:
        raise ValueError("max_presample_rows must be positive")
    missing = {"id1", "id2", "target"} - set(weak.columns)
    if missing:
        raise ValueError(f"weak pairs missing columns: {sorted(missing)}")
    out = weak.loc[:, ["id1", "id2", "target"]].copy()
    out["weak_weight"] = _weak_weight_vector(out["target"])
    keep = out["weak_weight"].to_numpy() > 0
    if validation_item_ids:
        keep &= ~out["id1"].isin(validation_item_ids).to_numpy()
        keep &= ~out["id2"].isin(validation_item_ids).to_numpy()
    out = out.loc[keep].reset_index(drop=True)
    if len(out) > max_presample_rows:
        out = out.sample(n=max_presample_rows, random_state=seed).reset_index(drop=True)
    out["hard_target"] = (out["target"].astype(float) >= 0.5).astype(np.int8)
    return out


def _weak_prefilter_mask_arrow(
    batch: pa.RecordBatch,
    validation_item_ids: set[object],
) -> pa.BooleanArray:
    target_index = batch.schema.get_field_index("target")
    target = pc.cast(batch.column(target_index), pa.float64())
    out_of_range = pc.or_(pc.less(target, 0.0), pc.greater(target, 1.0))
    if int(pc.sum(pc.cast(pc.fill_null(out_of_range, False), pa.int64())).as_py() or 0):
        raise ValueError("weak target must be in [0,1]")
    keep = pc.or_(pc.less_equal(target, 0.30), pc.greater_equal(target, 0.70))
    if validation_item_ids:
        for name in ("id1", "id2"):
            ids = batch.column(batch.schema.get_field_index(name))
            values = pa.array(list(validation_item_ids), type=ids.type)
            keep = pc.and_(keep, pc.invert(pc.is_in(ids, value_set=values)))
    return pc.fill_null(keep, False)


def prefilter_weak_candidates_parquet(
    path: Path,
    validation_item_ids: set[object],
    max_presample_rows: int,
    seed: int = SEED,
    *,
    batch_size: int = 250_000,
) -> tuple[pd.DataFrame, int]:
    """Reproduce ``prefilter_weak_candidates`` without loading the full parquet.

    Sampling is performed over the ordinal positions of eligible rows using the
    same ``RandomState.choice`` operation that backs pandas ``DataFrame.sample``.
    A second streaming pass materializes only the selected rows and then restores
    the original random-choice order, making this path deterministic and
    equivalent to the retained in-memory recipe while bounding peak RAM.
    """
    if max_presample_rows <= 0:
        raise ValueError("max_presample_rows must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    columns = ["id1", "id2", "target"]
    parquet = pq.ParquetFile(str(path))
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"weak pairs missing columns: {sorted(missing)}")
    input_rows = int(parquet.metadata.num_rows)

    eligible_count = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        keep = _weak_prefilter_mask_arrow(batch, validation_item_ids)
        eligible_count += int(pc.sum(pc.cast(keep, pa.int64())).as_py() or 0)

    if eligible_count > max_presample_rows:
        selected_ordinals = np.random.RandomState(seed).choice(
            eligible_count,
            size=max_presample_rows,
            replace=False,
        )
    else:
        selected_ordinals = np.arange(eligible_count, dtype=np.int64)

    if len(selected_ordinals) == 0:
        empty = pd.DataFrame(columns=columns)
        empty["weak_weight"] = pd.Series(dtype=np.float32)
        empty["hard_target"] = pd.Series(dtype=np.int8)
        return empty, input_rows

    selected_ordinals = np.asarray(selected_ordinals, dtype=np.int64)
    sorted_order = np.argsort(selected_ordinals, kind="stable")
    selected_sorted = selected_ordinals[sorted_order]
    pieces: list[pd.DataFrame] = []
    eligible_cursor = 0
    selected_cursor = 0

    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        keep = _weak_prefilter_mask_arrow(batch, validation_item_ids)
        eligible_positions = np.flatnonzero(keep.to_numpy(zero_copy_only=False))
        next_eligible_cursor = eligible_cursor + len(eligible_positions)
        selected_end = int(
            np.searchsorted(selected_sorted, next_eligible_cursor, side="left")
        )
        if selected_end > selected_cursor:
            ordinals = selected_sorted[selected_cursor:selected_end]
            local_offsets = ordinals - eligible_cursor
            raw_positions = eligible_positions[local_offsets]
            piece = batch.take(pa.array(raw_positions, type=pa.int64())).to_pandas()
            piece["_sample_order"] = sorted_order[selected_cursor:selected_end]
            pieces.append(piece)
        selected_cursor = selected_end
        eligible_cursor = next_eligible_cursor

    if eligible_cursor != eligible_count or selected_cursor != len(selected_ordinals):
        raise RuntimeError("weak parquet changed during deterministic sampling")

    out = pd.concat(pieces, ignore_index=True)
    out = (
        out.sort_values("_sample_order", kind="stable")
        .drop(columns="_sample_order")
        .reset_index(drop=True)
    )
    out["weak_weight"] = _weak_weight_vector(out["target"])
    out["hard_target"] = (out["target"].astype(float) >= 0.5).astype(np.int8)
    return out, input_rows


def candidate_sample_weights(
    categories: pd.Series,
    source: pd.Series,
    targets,
    weak_weight,
    hard_score,
    hard_negative_boost: float = 0.0,
) -> np.ndarray:
    categories = categories.reset_index(drop=True).astype(str)
    source = source.reset_index(drop=True).astype(str)
    target = np.asarray(targets, dtype=np.int8)
    weak = np.asarray(weak_weight, dtype=float)
    hard = np.asarray(hard_score, dtype=float)
    n = len(categories)
    if not (len(source) == len(target) == len(weak) == len(hard) == n):
        raise ValueError("weight inputs must have the same length")
    if hard_negative_boost < 0:
        raise ValueError("hard_negative_boost must be non-negative")

    category_weight = category_equalizing_weights(categories)
    base = np.where(source.to_numpy() == "human", 10.0, weak)
    negative_boost = np.where(target == 0, 1.0 + hard_negative_boost * np.clip(hard, 0.0, 1.0), 1.0)
    weights = category_weight * base * negative_boost
    # Keep the mean near one for numerically stable tree regularization while
    # preserving all relative human/weak/category/hard-negative ratios.
    mean = float(weights.mean()) if len(weights) else 1.0
    return weights / mean if mean > 0 else weights


def _validation_item_ids(matches: pd.DataFrame, valid_idx: np.ndarray) -> set[object]:
    valid = matches.iloc[valid_idx]
    return set(valid["id1"]) | set(valid["id2"])


def _score_candidate(
    name: str,
    model: Pipeline,
    x_valid: pd.DataFrame,
    valid_pairs: pd.DataFrame,
    fit_seconds: float,
) -> tuple[dict[str, Any], np.ndarray]:
    score = model.predict_proba(x_valid)[:, 1]
    if not np.isfinite(score).all():
        raise RuntimeError(f"{name} validation predictions contain NaN/Inf")
    macro_ap, per_cat = macro_average_precision(
        valid_pairs["target"].to_numpy(), score, valid_pairs["category"].to_numpy()
    )
    return {
        "candidate": name,
        "macro_average_precision": float(macro_ap),
        "per_category_ap": per_cat,
        "fit_seconds": float(fit_seconds),
    }, score


def _human_training_frame(
    human_matches: pd.DataFrame,
    human_items: pd.DataFrame,
    outer_train_idx: np.ndarray,
    transitive_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw = human_matches.iloc[outer_train_idx][["id1", "id2", "target"]].reset_index(drop=True)
    clean, clean_report = clean_human_pairs(raw)
    augmented, graph_report = augment_transitive_positives(clean, max_pairs_per_component=transitive_cap)
    augmented = attach_pair_category(augmented[["id1", "id2", "target"]], human_items)
    clean_with_category = attach_pair_category(clean[["id1", "id2", "target"]], human_items)
    return augmented, clean_with_category, {"clean": clean_report, "graph": graph_report}


def render_metrics_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# E-CUP v2 structured ablations",
        "",
        f"- v1 anchor Macro AP: **{payload['v1_anchor_macro_ap']:.6f}**",
        f"- fixed validation rows: {payload['validation_rows']:,}",
        f"- validation item overlap: {payload['validation_item_overlap']}",
        f"- selected structured candidate: **{payload['selected_candidate']}**",
        f"- selected Macro AP: **{payload['selected_macro_average_precision']:.6f}**",
        "",
    ]
    for candidate in payload["candidates"]:
        lines += [
            f"## {candidate['candidate']}",
            "",
            f"Macro AP: **{candidate['macro_average_precision']:.6f}**",
            "",
            "| category | AP |",
            "|---|---:|",
        ]
        for category, value in candidate["per_category_ap"].items():
            lines.append(f"| {category} | {value:.6f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def train_structured_ablation(
    human_items_path: Path,
    human_matches_path: Path,
    llm_matches_path: Path | None,
    full_items_path: Path | None,
    output_dir: Path,
    weak_presample_rows: int = 450_000,
    weak_final_rows: int = 300_000,
    transitive_cap: int = 2000,
    chunk_size: int = 25_000,
    max_iter: int = 350,
    hard_negative_boost: float = 3.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    human_items = pd.read_parquet(human_items_path, columns=["id", "name", "attributes", "category"])
    human_matches = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    human_matches = attach_pair_category(human_matches, human_items)

    train_idx, valid_idx = fixed_v1_split(human_matches)
    train_items = set(human_matches.iloc[train_idx]["id1"]) | set(human_matches.iloc[train_idx]["id2"])
    valid_items = _validation_item_ids(human_matches, valid_idx)
    overlap = len(train_items & valid_items)
    if overlap:
        raise RuntimeError(f"item leakage in fixed split: {overlap}")

    valid_pairs = human_matches.iloc[valid_idx].reset_index(drop=True)
    human_aug, human_clean, graph_report = _human_training_frame(
        human_matches, human_items, train_idx, transitive_cap
    )
    importance = learn_attribute_importance(human_items, human_clean, min_support=20)

    feature_started = time.perf_counter()
    x_human = build_features_v2_chunked(
        human_items, human_aug, attribute_importance=importance, chunk_size=chunk_size
    )
    x_valid = build_features_v2_chunked(
        human_items, valid_pairs, attribute_importance=importance, chunk_size=chunk_size
    )
    human_feature_seconds = time.perf_counter() - feature_started

    human_source = pd.Series(["human"] * len(human_aug))
    human_weak_weight = np.ones(len(human_aug), dtype=float)
    human_hard = x_human["hard_negative_score"].to_numpy(float)
    w_a = candidate_sample_weights(
        human_aug["category"], human_source, human_aug["target"].to_numpy(),
        human_weak_weight, human_hard, hard_negative_boost=0.0,
    )
    fit_started = time.perf_counter()
    model_a = fit_estimator(x_human, human_aug["target"].to_numpy(), w_a, max_iter=max_iter)
    metric_a, pred_a = _score_candidate("v2a-human-2024-features", model_a, x_valid, valid_pairs, time.perf_counter() - fit_started)

    candidates: list[dict[str, Any]] = [metric_a]
    models: dict[str, Pipeline] = {metric_a["candidate"]: model_a}
    predictions: dict[str, np.ndarray] = {metric_a["candidate"]: pred_a}
    weak_report: dict[str, Any] = {"enabled": False}
    x_weak: pd.DataFrame | None = None
    weak_pairs: pd.DataFrame | None = None

    if llm_matches_path is not None and full_items_path is not None:
        weak_started = time.perf_counter()
        weak, input_weak_rows = prefilter_weak_candidates_parquet(
            llm_matches_path,
            validation_item_ids=valid_items,
            max_presample_rows=weak_presample_rows,
            seed=SEED,
        )
        weak, prep_report = prepare_weak_pairs(weak[["id1", "id2", "target"]])
        weak, conflict_report = remove_human_conflicts(
            weak,
            human_clean[["id1", "id2", "target"]],
        )
        weak_ids = set(weak["id1"]) | set(weak["id2"])
        weak_items = select_items_by_ids(full_items_path, weak_ids)
        weak = attach_pair_category(weak, weak_items)
        weak = sample_weak_training(weak, max_rows=weak_final_rows, seed=SEED)
        # Item subset can now be reduced to the final sample before feature extraction.
        final_weak_ids = set(weak["id1"]) | set(weak["id2"])
        weak_items = weak_items[weak_items["id"].isin(final_weak_ids)].reset_index(drop=True)
        x_weak = build_features_v2_chunked(
            weak_items, weak, attribute_importance=importance, chunk_size=chunk_size
        )
        weak_pairs = weak.reset_index(drop=True)
        weak_feature_seconds = time.perf_counter() - weak_started
        weak_report = {
            "enabled": True,
            "input_rows": input_weak_rows,
            "after_prefilter_rows": int(min(weak_presample_rows, input_weak_rows)),
            "final_rows": int(len(weak_pairs)),
            "unique_items": int(len(final_weak_ids)),
            "prepare": prep_report,
            "conflicts": conflict_report,
            "feature_and_selection_seconds": float(weak_feature_seconds),
        }

        x_combined = pd.concat([x_human, x_weak], ignore_index=True)
        y_combined = np.concatenate([
            human_aug["target"].to_numpy(np.int8),
            weak_pairs["hard_target"].to_numpy(np.int8),
        ])
        category_combined = pd.concat([
            human_aug["category"].reset_index(drop=True),
            weak_pairs["category"].reset_index(drop=True),
        ], ignore_index=True)
        source_combined = pd.Series(
            ["human"] * len(human_aug) + ["weak"] * len(weak_pairs)
        )
        weak_weight_combined = np.concatenate([
            np.ones(len(human_aug), dtype=float),
            weak_pairs["weak_weight"].to_numpy(float),
        ])
        hard_combined = x_combined["hard_negative_score"].to_numpy(float)

        w_b = candidate_sample_weights(
            category_combined, source_combined, y_combined, weak_weight_combined,
            hard_combined, hard_negative_boost=0.0,
        )
        fit_started = time.perf_counter()
        model_b = fit_estimator(x_combined, y_combined, w_b, max_iter=max_iter)
        metric_b, pred_b = _score_candidate("v2b-weak-curriculum", model_b, x_valid, valid_pairs, time.perf_counter() - fit_started)
        candidates.append(metric_b)
        models[metric_b["candidate"]] = model_b
        predictions[metric_b["candidate"]] = pred_b

        w_c = candidate_sample_weights(
            category_combined, source_combined, y_combined, weak_weight_combined,
            hard_combined, hard_negative_boost=hard_negative_boost,
        )
        fit_started = time.perf_counter()
        model_c = fit_estimator(x_combined, y_combined, w_c, max_iter=max_iter)
        metric_c, pred_c = _score_candidate("v2c-hard-negative-weighting", model_c, x_valid, valid_pairs, time.perf_counter() - fit_started)
        candidates.append(metric_c)
        models[metric_c["candidate"]] = model_c
        predictions[metric_c["candidate"]] = pred_c

    selected = max(candidates, key=lambda item: item["macro_average_precision"])
    selected_name = selected["candidate"]

    # Refit the selected structured recipe with the original fixed-validation
    # human rows added back. We deliberately retain the train-derived attribute
    # importance so feature definitions do not change after model selection.
    x_all_human = pd.concat([x_human, x_valid], ignore_index=True)
    all_human_pairs = pd.concat([human_aug, valid_pairs], ignore_index=True)
    if selected_name == "v2a-human-2024-features" or x_weak is None or weak_pairs is None:
        x_final = x_all_human
        y_final = all_human_pairs["target"].to_numpy(np.int8)
        final_categories = all_human_pairs["category"].reset_index(drop=True)
        final_source = pd.Series(["human"] * len(all_human_pairs))
        final_weak_weight = np.ones(len(all_human_pairs), dtype=float)
        final_hard_boost = 0.0
    else:
        x_final = pd.concat([x_all_human, x_weak], ignore_index=True)
        y_final = np.concatenate([
            all_human_pairs["target"].to_numpy(np.int8),
            weak_pairs["hard_target"].to_numpy(np.int8),
        ])
        final_categories = pd.concat([
            all_human_pairs["category"].reset_index(drop=True),
            weak_pairs["category"].reset_index(drop=True),
        ], ignore_index=True)
        final_source = pd.Series(["human"] * len(all_human_pairs) + ["weak"] * len(weak_pairs))
        final_weak_weight = np.concatenate([
            np.ones(len(all_human_pairs), dtype=float),
            weak_pairs["weak_weight"].to_numpy(float),
        ])
        final_hard_boost = hard_negative_boost if selected_name.startswith("v2c") else 0.0
    final_weights = candidate_sample_weights(
        final_categories, final_source, y_final, final_weak_weight,
        x_final["hard_negative_score"].to_numpy(float),
        hard_negative_boost=final_hard_boost,
    )
    final_fit_started = time.perf_counter()
    final_model = fit_estimator(x_final, y_final, final_weights, max_iter=max_iter)
    final_fit_seconds = time.perf_counter() - final_fit_started

    payload: dict[str, Any] = {
        "version": "v2-2024-transfer-structured",
        "seed": SEED,
        "v1_anchor_macro_ap": V1_MACRO_AP,
        "selected_candidate": selected_name,
        "selected_macro_average_precision": float(selected["macro_average_precision"]),
        "delta_vs_v1": float(selected["macro_average_precision"] - V1_MACRO_AP),
        "accepted_as_improvement": bool(selected["macro_average_precision"] > V1_MACRO_AP),
        "candidates": candidates,
        "validation_rows": int(len(valid_pairs)),
        "validation_item_overlap": int(overlap),
        "outer_train_rows": int(len(human_matches.iloc[train_idx])),
        "human_augmented_rows": int(len(human_aug)),
        "human_feature_seconds": float(human_feature_seconds),
        "graph_report": graph_report,
        "weak_report": weak_report,
        "attribute_importance_categories": int(len(importance)),
        "final_fit_rows": int(len(x_final)),
        "final_fit_seconds": float(final_fit_seconds),
        "total_seconds": float(time.perf_counter() - started),
        "feature_names": list(FEATURE_NAMES_V2),
        "hard_negative_boost": float(hard_negative_boost),
    }
    manifest = {
        "version": "v2-2024-transfer-structured",
        "selected_candidate": selected_name,
        "validation_macro_ap": float(selected["macro_average_precision"]),
        "feature_names": list(FEATURE_NAMES_V2),
        "attribute_importance": importance,
        "hard_negative_boost": float(final_hard_boost),
        "seed": SEED,
        "model_class": "sklearn.ensemble.HistGradientBoostingClassifier",
    }
    save_model_bundle(
        final_model,
        output_dir / "model.joblib",
        output_dir / "manifest.json",
        manifest,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.md").write_text(render_metrics_markdown(payload), encoding="utf-8")

    pred = valid_pairs[["id1", "id2", "target", "category"]].copy()
    for name, values in predictions.items():
        pred[name] = values
    pred.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-items", required=True, type=Path)
    parser.add_argument("--human-matches", required=True, type=Path)
    parser.add_argument("--llm-matches", type=Path)
    parser.add_argument("--full-items", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weak-presample-rows", type=int, default=450_000)
    parser.add_argument("--weak-final-rows", type=int, default=300_000)
    parser.add_argument("--transitive-cap", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument("--hard-negative-boost", type=float, default=3.0)
    args = parser.parse_args()
    metrics = train_structured_ablation(
        args.human_items,
        args.human_matches,
        args.llm_matches,
        args.full_items,
        args.output_dir,
        weak_presample_rows=args.weak_presample_rows,
        weak_final_rows=args.weak_final_rows,
        transitive_cap=args.transitive_cap,
        chunk_size=args.chunk_size,
        max_iter=args.max_iter,
        hard_negative_boost=args.hard_negative_boost,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
