from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .category_attrs import learn_attribute_importance
from .label_graph import clean_human_pairs
from .reranker_data import build_reranker_examples
from .train_reranker_v2 import _soft_category_weights, _train_model
from .train_v1 import attach_pair_category
from .train_v4_reranker import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    SEED,
    _load_local_model,
    _verify_model_revision,
)
from .v2_split import fixed_v1_split


def sample_benchmark_pairs(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    seed: int = SEED,
) -> pd.DataFrame:
    required = {"id1", "id2", "target", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"benchmark pairs missing columns: {sorted(missing)}")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    work = frame.copy().reset_index(drop=True)
    if len(work) <= max_rows:
        return work.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    groups = list(work.groupby(["category", "target"], sort=True, dropna=False))
    if not groups:
        raise ValueError("benchmark source is empty")
    quota = max(1, max_rows // len(groups))
    selected: list[pd.DataFrame] = []
    used: set[int] = set()
    for number, (_, group) in enumerate(groups):
        take = min(quota, len(group))
        sample = group.sample(n=take, random_state=seed + number, replace=False)
        selected.append(sample)
        used.update(sample.index.tolist())
    out = pd.concat(selected, axis=0) if selected else work.iloc[:0].copy()
    remaining = max_rows - len(out)
    if remaining > 0:
        pool = work.loc[~work.index.isin(used)]
        if len(pool):
            fill = pool.sample(
                n=min(remaining, len(pool)),
                random_state=seed + 10_000,
                replace=False,
            )
            out = pd.concat([out, fill], axis=0)
    if len(out) != max_rows:
        raise RuntimeError(f"expected {max_rows} benchmark rows, got {len(out)}")
    return out.sample(frac=1.0, random_state=seed + 20_000).reset_index(drop=True)


def prepare_benchmark_examples(
    human_items_path: Path,
    human_matches_path: Path,
    *,
    benchmark_rows: int,
    max_attrs: int,
    max_chars: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    items = pd.read_parquet(
        human_items_path, columns=["id", "name", "attributes", "category"]
    )
    matches = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    matches = attach_pair_category(matches, items)
    train_idx, valid_idx = fixed_v1_split(matches)
    outer_train = matches.iloc[train_idx].reset_index(drop=True)
    validation = matches.iloc[valid_idx].reset_index(drop=True)
    validation_ids = set(validation["id1"]) | set(validation["id2"])
    train_ids = set(outer_train["id1"]) | set(outer_train["id2"])
    overlap = train_ids & validation_ids
    if overlap:
        raise RuntimeError(f"fixed split has {len(overlap)} overlapping item IDs")

    clean, clean_report = clean_human_pairs(outer_train[["id1", "id2", "target"]])
    clean = attach_pair_category(clean, items)
    sample = sample_benchmark_pairs(clean, max_rows=benchmark_rows, seed=SEED)
    sample_ids = set(sample["id1"]) | set(sample["id2"])
    sample_overlap = sample_ids & validation_ids
    if sample_overlap:
        raise RuntimeError(
            f"benchmark sample leaks {len(sample_overlap)} validation item IDs"
        )
    importance = learn_attribute_importance(items, clean, min_support=20)
    source = pd.Series(["human"] * len(sample), dtype=object)
    weights = _soft_category_weights(
        sample["category"].astype(str).reset_index(drop=True),
        source,
        np.ones(len(sample), dtype=float),
    )
    sample = sample[["id1", "id2", "target", "category"]].copy()
    sample["sample_weight"] = weights
    sample["source"] = "human"
    examples = build_reranker_examples(
        items,
        sample,
        importance,
        max_attrs=max_attrs,
        max_chars=max_chars,
    )
    report: dict[str, object] = {
        "human_rows": int(len(matches)),
        "outer_train_rows": int(len(outer_train)),
        "validation_rows": int(len(validation)),
        "validation_item_overlap": 0,
        "clean_rows": int(len(clean)),
        "benchmark_rows": int(len(examples)),
        "benchmark_positive_rows": int((examples["target"].astype(float) >= 0.5).sum()),
        "benchmark_negative_rows": int((examples["target"].astype(float) < 0.5).sum()),
        "benchmark_categories": int(examples["category"].astype(str).nunique()),
        "clean_report": clean_report,
    }
    return examples, report


def benchmark_v4_cuda(
    *,
    human_items_path: Path,
    human_matches_path: Path,
    output_dir: Path,
    model_path: str,
    base_model_revision: str,
    benchmark_rows: int = 2_048,
    max_length: int = 256,
    train_batch_size: int = 2,
    gradient_accumulation: int = 16,
    gradient_checkpointing: bool = False,
    benchmark_epochs: float = 0.25,
    max_attrs: int = 10,
    max_chars: int = 700,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, data_report = prepare_benchmark_examples(
        human_items_path,
        human_matches_path,
        benchmark_rows=benchmark_rows,
        max_attrs=max_attrs,
        max_chars=max_chars,
    )
    _verify_model_revision(model_path, base_model_revision)
    torch, tokenizer, model = _load_local_model(
        model_path,
        gradient_checkpointing=gradient_checkpointing,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    training = _train_model(
        model,
        tokenizer,
        frame,
        batch_size=train_batch_size,
        epochs=benchmark_epochs,
        learning_rate=1.5e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        max_length=max_length,
        gradient_accumulation=gradient_accumulation,
        seed=SEED,
    )
    seconds = time.perf_counter() - started
    examples_seen = int(training["batches"]) * int(train_batch_size)
    payload: dict[str, object] = {
        "version": "v4-cuda-benchmark",
        "base_model": DEFAULT_MODEL,
        "base_model_revision": base_model_revision.lower(),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": str(torch.__version__),
        "max_length": int(max_length),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation": int(gradient_accumulation),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "benchmark_rows": int(len(frame)),
        "examples_seen": int(examples_seen),
        "seconds": float(seconds),
        "examples_per_second": float(examples_seen / max(seconds, 1e-9)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "data_report": data_report,
        "training": training,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-items", required=True, type=Path)
    parser.add_argument("--human-matches", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="/opt/models/rubert-base")
    parser.add_argument("--base-model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--benchmark-rows", type=int, default=2_048)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--benchmark-epochs", type=float, default=0.25)
    parser.add_argument("--max-attrs", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=700)
    args = parser.parse_args()
    benchmark_v4_cuda(
        human_items_path=args.human_items,
        human_matches_path=args.human_matches,
        output_dir=args.output_dir,
        model_path=args.base_model,
        base_model_revision=args.base_model_revision,
        benchmark_rows=args.benchmark_rows,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        gradient_checkpointing=args.gradient_checkpointing,
        benchmark_epochs=args.benchmark_epochs,
        max_attrs=args.max_attrs,
        max_chars=args.max_chars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
