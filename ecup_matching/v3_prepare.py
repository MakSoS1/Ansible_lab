from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

from .ml.train_reranker_v2 import prepare_training_examples


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

    compact = compact_serialized_examples(
        train,
        max_rows=min(max_train_rows, len(train)),
        priority_categories=PRIORITY_CATEGORIES,
        priority_fraction=priority_fraction,
        seed=SEED,
    )
    valid = valid.copy()
    valid["source"] = "human"

    valid_ids = set(valid["id1"]) | set(valid["id2"])
    train_ids = set(compact["id1"]) | set(compact["id2"])
    overlap = train_ids & valid_ids
    if overlap:
        raise RuntimeError(f"prepared v3 data leaks {len(overlap)} validation item IDs")

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
        "seed": SEED,
        "train_rows_before_compaction": int(len(train)),
        "train_rows": int(len(compact)),
        "validation_rows": int(len(valid)),
        "validation_item_overlap": 0,
        "human_positive_rows_retained": int(
            ((compact["source"] == "human") & (compact["target"].astype(float) >= 0.5)).sum()
        ),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-items", type=Path, required=True)
    parser.add_argument("--human-matches", type=Path, required=True)
    parser.add_argument("--llm-matches", type=Path, required=True)
    parser.add_argument("--full-items", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=180_000)
    parser.add_argument("--weak-presample-rows", type=int, default=350_000)
    parser.add_argument("--weak-final-rows", type=int, default=220_000)
    parser.add_argument("--max-attrs", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=700)
    parser.add_argument("--priority-fraction", type=float, default=0.60)
    args = parser.parse_args()
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
