from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark_v4_cuda import sample_benchmark_pairs
from .train_v1 import attach_pair_category
from .train_v4_reranker import DEFAULT_MODEL_REVISION, _verify_model_revision
from .v7_neural import build_v7_text_cache, configure_trainable_layers, predict_pairs, train_pair_phase


def prepare_v7_benchmark(
    human_items_path: Path,
    human_matches_path: Path,
    *,
    benchmark_rows: int,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = pd.read_parquet(
        human_items_path, columns=["id", "name", "attributes", "category"]
    )
    matches = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    matches = attach_pair_category(matches, items)
    sample = sample_benchmark_pairs(matches, max_rows=benchmark_rows, seed=seed)
    needed = set(sample["id1"].tolist()) | set(sample["id2"].tolist())
    item_subset = items[items["id"].isin(needed)].copy().reset_index(drop=True)
    if set(item_subset["id"].tolist()) != needed:
        raise RuntimeError("benchmark item subset is incomplete")
    return item_subset, sample[["id1", "id2", "target", "category"]].reset_index(drop=True)


def benchmark_v7_cuda(
    *,
    human_items_path: Path,
    human_matches_path: Path,
    output_dir: Path,
    model_path: str,
    base_model_revision: str,
    max_length: int = 256,
    max_chars: int = 900,
    physical_batch_size: int = 2,
    benchmark_rows: int = 4096,
    warmup_rows: int = 256,
    seed: int = 2026,
) -> dict[str, object]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if max_length != 256:
        raise ValueError("v7 CUDA benchmark is pinned to max_length=256")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v7 benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    _verify_model_revision(model_path, base_model_revision)

    prepare_started = time.perf_counter()
    items, frame = prepare_v7_benchmark(
        human_items_path,
        human_matches_path,
        benchmark_rows=benchmark_rows,
        seed=seed,
    )
    texts = build_v7_text_cache(items, max_chars=max_chars)
    prepare_seconds = time.perf_counter() - prepare_started

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        num_labels=1,
        ignore_mismatched_sizes=True,
    ).to("cuda")
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    configure_trainable_layers(model, last_n_layers=8)

    warm = frame.iloc[: min(len(frame), warmup_rows)].reset_index(drop=True)
    if len(warm):
        predict_pairs(
            model=model,
            tokenizer=tokenizer,
            frame=warm,
            texts=texts,
            device="cuda",
            max_length=max_length,
            batch_size=16,
        )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _, inference = predict_pairs(
        model=model,
        tokenizer=tokenizer,
        frame=frame,
        texts=texts,
        device="cuda",
        max_length=max_length,
        batch_size=16,
    )

    training = train_pair_phase(
        model=model,
        tokenizer=tokenizer,
        frame=frame,
        texts=texts,
        device="cuda",
        phase="v7-benchmark-train-check",
        epochs=0.03,
        physical_batch_size=physical_batch_size,
        effective_batch_size=32,
        max_length=max_length,
        learning_rate=1.5e-5,
        ranking_weight=0.25,
        seed=seed,
        weak=False,
        telemetry_every_steps=20,
    )

    examples_per_second = float(inference["examples_per_second"])
    payload: dict[str, object] = {
        "version": "v7-cuda-benchmark",
        "base_model": "ai-forever/ruBert-base",
        "base_model_revision": base_model_revision.lower(),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": str(torch.__version__),
        "max_length": int(max_length),
        "max_chars": int(max_chars),
        "physical_batch_size": int(physical_batch_size),
        "benchmark_rows": int(len(frame)),
        "warmup_rows": int(len(warm)),
        "examples_seen": int(len(frame)),
        "seconds": float(inference["seconds"]),
        "examples_per_second": examples_per_second,
        "peak_allocated_bytes": int(inference["peak_allocated_bytes"]),
        "peak_reserved_bytes": int(inference["peak_reserved_bytes"]),
        "inference": inference,
        "training_check": training.__dict__,
        "prepare_seconds": float(prepare_seconds),
        "projected_public_neural_seconds_115k_on_this_gpu": float(115_000 / max(examples_per_second, 1e-9)),
        "projected_private_neural_seconds_275k_on_this_gpu": float(275_000 / max(examples_per_second, 1e-9)),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-items", type=Path, required=True)
    parser.add_argument("--human-matches", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="/opt/models/rubert-base")
    parser.add_argument("--base-model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--physical-batch-size", type=int, default=2)
    parser.add_argument("--benchmark-rows", type=int, default=4096)
    parser.add_argument("--warmup-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    benchmark_v7_cuda(
        human_items_path=args.human_items,
        human_matches_path=args.human_matches,
        output_dir=args.output_dir,
        model_path=args.base_model,
        base_model_revision=args.base_model_revision,
        max_length=args.max_length,
        max_chars=args.max_chars,
        physical_batch_size=args.physical_batch_size,
        benchmark_rows=args.benchmark_rows,
        warmup_rows=args.warmup_rows,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())