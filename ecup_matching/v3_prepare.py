from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .ml.category_attrs import learn_attribute_importance
from .ml.label_graph import augment_transitive_positives, clean_human_pairs
from .ml.reranker_data import build_reranker_examples
from .ml.train_reranker_v2 import _soft_category_weights, prepare_training_examples
from .ml.train_v1 import attach_pair_category
from .ml.v2_split import fixed_v1_split


SEED = 2026
PRIORITY_CATEGORIES = {
    "Электроника",
    "Одежда",
    "Обувь",
    "Ювелирные изделия",
    "Галантерея и аксессуары",
    "Мебель",
}


def prepared_remote_prefix(commit_sha: str) -> str:
    sha = str(commit_sha).strip()
    if len(sha) < 12 or not all(ch in "0123456789abcdefABCDEF" for ch in sha):
        raise ValueError("commit_sha must contain at least 12 hexadecimal characters")
    return f"experiments/v3/prepared/{sha[:12].lower()}"


def _sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[:0].copy()
    if len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed)


def compact_serialized_examples(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    priority_categories: set[str] = PRIORITY_CATEGORIES,
    priority_fraction: float = 0.60,
    seed: int = SEED,
) -> pd.DataFrame:
    required = {
        "id1",
        "id2",
        "target",
        "category",
        "sample_weight",
        "text_a",
        "text_b",
        "source",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"serialized examples missing columns: {sorted(missing)}")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if not 0.0 <= priority_fraction <= 1.0:
        raise ValueError("priority_fraction must be in [0,1]")

    work = frame.copy().reset_index(drop=True)
    mandatory_mask = (work["source"].astype(str) == "human") & (
        pd.to_numeric(work["target"], errors="raise").astype(float) >= 0.5
    )
    mandatory = work.loc[mandatory_mask].copy()
    if len(mandatory) > max_rows:
        raise ValueError(
            f"max_rows={max_rows} cannot preserve {len(mandatory)} human positives"
        )
    discretionary = work.loc[~mandatory_mask].copy()
    budget = max_rows - len(mandatory)
    if len(discretionary) < budget:
        return work.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    priority_mask = discretionary["category"].astype(str).isin(priority_categories)
    priority = discretionary.loc[priority_mask]
    regular = discretionary.loc[~priority_mask]
    priority_n = min(math.ceil(budget * priority_fraction), len(priority))
    picked_priority = _sample(priority, priority_n, seed)
    remaining = budget - len(picked_priority)
    picked_regular = _sample(regular, min(remaining, len(regular)), seed + 1)
    remaining -= len(picked_regular)

    if remaining:
        used = set(picked_priority.index) | set(picked_regular.index)
        leftovers = discretionary.loc[~discretionary.index.isin(used)]
        fill = _sample(leftovers, remaining, seed + 2)
    else:
        fill = discretionary.iloc[:0].copy()

    result = pd.concat([mandatory, picked_priority, picked_regular, fill], ignore_index=True)
    if len(result) != max_rows:
        raise RuntimeError(f"expected {max_rows} compact rows, got {len(result)}")
    return result.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)


def _write_prepared_outputs(
    *,
    train_examples: pd.DataFrame,
    valid_examples: pd.DataFrame,
    output_dir: Path,
    report: dict[str, object],
    mode: str,
    max_train_rows: int,
    priority_fraction: float,
    max_attrs: int,
    max_chars: int,
) -> dict[str, object]:
    compact = compact_serialized_examples(
        train_examples,
        max_rows=min(max_train_rows, len(train_examples)),
        priority_categories=PRIORITY_CATEGORIES,
        priority_fraction=priority_fraction,
        seed=SEED,
    )
    valid = valid_examples.copy()
    valid["source"] = "human"

    valid_ids = set(valid["id1"]) | set(valid["id2"])
    train_ids = set(compact["id1"]) | set(compact["id2"])
    overlap = train_ids & valid_ids
    if overlap:
        raise RuntimeError(f"prepared v3 data leaks {len(overlap)} validation item IDs")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_examples.parquet"
    valid_path = output_dir / "validation_examples.parquet"
    compact.to_parquet(train_path, index=False)
    valid.to_parquet(valid_path, index=False)

    source_counts = {
        str(k): int(v) for k, v in compact["source"].value_counts().to_dict().items()
    }
    priority_counts = {
        str(k): int(v)
        for k, v in compact.loc[
            compact["category"].astype(str).isin(PRIORITY_CATEGORIES), "category"
        ].value_counts().to_dict().items()
    }
    payload: dict[str, object] = {
        "version": "v3-prepared-neural-data",
        "mode": mode,
        "seed": SEED,
        "train_rows_before_compaction": int(len(train_examples)),
        "train_rows": int(len(compact)),
        "validation_rows": int(len(valid)),
        "validation_item_overlap": 0,
        "human_positive_rows_retained": int(
            ((compact["source"] == "human") & (compact["target"].astype(float) >= 0.5)).sum()
        ),
        "weak_rows": int((compact["source"].astype(str) == "weak").sum()),
        "source_counts": source_counts,
        "priority_category_counts": priority_counts,
        "priority_fraction": float(priority_fraction),
        "max_attrs": int(max_attrs),
        "max_chars": int(max_chars),
        "upstream_report": report,
        "files": {
            "train_examples": train_path.name,
            "validation_examples": valid_path.name,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def prepare_v3_human_only_data(
    *,
    human_items_path: Path,
    human_matches_path: Path,
    output_dir: Path,
    max_train_rows: int = 180_000,
    max_attrs: int = 10,
    max_chars: int = 700,
    priority_fraction: float = 0.60,
) -> dict[str, object]:
    """Prepare the first v3 neural curriculum using authoritative human data only.

    This intentionally avoids the 4.1 GiB full item table. The fixed v1/v2
    outer split remains unchanged, train labels are cleaned only after the
    split, and validation stays untouched.
    """
    human_items = pd.read_parquet(
        human_items_path, columns=["id", "name", "attributes", "category"]
    )
    matches = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    matches = attach_pair_category(matches, human_items)
    train_idx, valid_idx = fixed_v1_split(matches)
    outer_train = matches.iloc[train_idx].reset_index(drop=True)
    valid = matches.iloc[valid_idx].reset_index(drop=True)
    valid_ids = set(valid["id1"]) | set(valid["id2"])
    outer_ids = set(outer_train["id1"]) | set(outer_train["id2"])
    overlap = valid_ids & outer_ids
    if overlap:
        raise RuntimeError(f"fixed split has {len(overlap)} overlapping item IDs")

    clean, clean_report = clean_human_pairs(outer_train[["id1", "id2", "target"]])
    augmented, graph_report = augment_transitive_positives(
        clean, max_pairs_per_component=1000
    )
    clean = attach_pair_category(clean, human_items)
    augmented = attach_pair_category(augmented, human_items)
    importance = learn_attribute_importance(human_items, clean, min_support=20)

    source = pd.Series(["human"] * len(augmented))
    weights = _soft_category_weights(
        augmented["category"].reset_index(drop=True),
        source,
        np.ones(len(augmented), dtype=float),
    )
    train_pairs = augmented[["id1", "id2", "target", "category"]].copy()
    train_pairs["sample_weight"] = weights
    train_pairs["source"] = "human"
    train_examples = build_reranker_examples(
        human_items,
        train_pairs,
        importance,
        max_attrs=max_attrs,
        max_chars=max_chars,
    )
    valid_pairs = valid[["id1", "id2", "target", "category"]].copy()
    valid_pairs["sample_weight"] = 1.0
    valid_pairs["source"] = "human"
    valid_examples = build_reranker_examples(
        human_items,
        valid_pairs,
        importance,
        max_attrs=max_attrs,
        max_chars=max_chars,
    )
    report: dict[str, object] = {
        "human_rows": int(len(matches)),
        "outer_train_rows": int(len(outer_train)),
        "validation_rows": int(len(valid)),
        "validation_item_overlap": 0,
        "human_clean_rows": int(len(clean)),
        "human_augmented_rows": int(len(augmented)),
        "weak_input_rows": 0,
        "weak_final_rows": 0,
        "attribute_importance": importance,
        "clean_report": clean_report,
        "graph_report": graph_report,
    }
    return _write_prepared_outputs(
        train_examples=train_examples,
        valid_examples=valid_examples,
        output_dir=output_dir,
        report=report,
        mode="human-only",
        max_train_rows=max_train_rows,
        priority_fraction=priority_fraction,
        max_attrs=max_attrs,
        max_chars=max_chars,
    )


def prepare_v3_private_data(
    *,
    human_items_path: Path,
    human_matches_path: Path,
    llm_matches_path: Path,
    full_items_path: Path,
    output_dir: Path,
    max_train_rows: int = 180_000,
    weak_presample_rows: int = 350_000,
    weak_final_rows: int = 220_000,
    max_attrs: int = 10,
    max_chars: int = 700,
    priority_fraction: float = 0.60,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, valid, report = prepare_training_examples(
        human_items_path,
        human_matches_path,
        llm_matches_path,
        full_items_path,
        weak_presample_rows=weak_presample_rows,
        weak_final_rows=weak_final_rows,
        transitive_cap=1000,
        max_attrs=max_attrs,
        max_chars=max_chars,
    )
    human_rows = int(report["human_augmented_rows"])
    if human_rows <= 0 or human_rows > len(train):
        raise RuntimeError("invalid human row boundary from v2 preprocessing report")
    train = train.copy()
    train["source"] = "weak"
    train.loc[: human_rows - 1, "source"] = "human"
    return _write_prepared_outputs(
        train_examples=train,
        valid_examples=valid,
        output_dir=output_dir,
        report=report,
        mode="human+weak",
        max_train_rows=max_train_rows,
        priority_fraction=priority_fraction,
        max_attrs=max_attrs,
        max_chars=max_chars,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-items", type=Path, required=True)
    parser.add_argument("--human-matches", type=Path, required=True)
    parser.add_argument("--llm-matches", type=Path)
    parser.add_argument("--full-items", type=Path)
    parser.add_argument("--human-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=180_000)
    parser.add_argument("--weak-presample-rows", type=int, default=350_000)
    parser.add_argument("--weak-final-rows", type=int, default=220_000)
    parser.add_argument("--max-attrs", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=700)
    parser.add_argument("--priority-fraction", type=float, default=0.60)
    args = parser.parse_args()
    if args.human_only:
        payload = prepare_v3_human_only_data(
            human_items_path=args.human_items,
            human_matches_path=args.human_matches,
            output_dir=args.output_dir,
            max_train_rows=args.max_train_rows,
            max_attrs=args.max_attrs,
            max_chars=args.max_chars,
            priority_fraction=args.priority_fraction,
        )
    else:
        if args.llm_matches is None or args.full_items is None:
            parser.error("--llm-matches and --full-items are required unless --human-only is used")
        payload = prepare_v3_private_data(
            human_items_path=args.human_items,
            human_matches_path=args.human_matches,
            llm_matches_path=args.llm_matches,
            full_items_path=args.full_items,
            output_dir=args.output_dir,
            max_train_rows=args.max_train_rows,
            weak_presample_rows=args.weak_presample_rows,
            weak_final_rows=args.weak_final_rows,
            max_attrs=args.max_attrs,
            max_chars=args.max_chars,
            priority_fraction=args.priority_fraction,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
