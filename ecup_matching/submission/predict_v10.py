"""v10 submission inference: one tiny pair cross-encoder, no heavyweight stages."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ecup_matching.ml.v7_runtime import (
    build_v7_text_cache_from_parquet,
    predict_pairs,
    serialization_workers,
)


DEFAULT_MAX_LENGTH = 128
DEFAULT_MAX_CHARS = 650
DEFAULT_BATCH_SIZE = 128
EXPECTED_BASE_MODEL = "cointegrated/rubert-tiny2"


def validate_v10_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(metadata)
    if payload.get("version") != "v10-tiny-student":
        raise ValueError("v10 metadata version must be v10-tiny-student")
    if payload.get("base_model") != EXPECTED_BASE_MODEL:
        raise ValueError("v10 runtime must use the pinned tiny student base model")
    if payload.get("gold_metric_opened") is not False or int(
        payload.get("gold_rows_scored", -1)
    ) != 0:
        raise ValueError("v10 metadata violates the sealed gold contract")

    strict = payload.get("strict_oof_macro_average_precision")
    if strict is not None and not 0.0 <= float(strict) <= 1.0:
        raise ValueError("strict_oof_macro_average_precision must be in [0,1]")

    max_length = int(payload.get("max_length", DEFAULT_MAX_LENGTH))
    max_chars = int(payload.get("max_chars", DEFAULT_MAX_CHARS))
    batch_size = int(payload.get("inference_batch_size", DEFAULT_BATCH_SIZE))
    if not 1 <= max_length <= 160:
        raise ValueError("v10 max_length must be within [1,160]")
    if not 1 <= max_chars <= 700:
        raise ValueError("v10 max_chars must be within [1,700]")
    if batch_size < 64:
        raise ValueError("v10 inference_batch_size must be at least 64")
    payload["max_length"] = max_length
    payload["max_chars"] = max_chars
    payload["inference_batch_size"] = batch_size
    return payload


def _phase(label: str, started: float, previous: float, **fields: Any) -> float:
    now = time.perf_counter()
    print(
        "[v10] "
        + json.dumps(
            {
                "phase": label,
                "seconds": round(now - previous, 3),
                "total_seconds": round(now - started, 3),
                **fields,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return now


def _select_device() -> str:
    try:
        import torch
    except ImportError:  # pragma: no cover - organizer image carries torch
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_dir: Path, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    kwargs: dict[str, Any] = {"local_files_only": True}
    if device == "cuda":
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir), attn_implementation="sdpa", **kwargs
            )
        except (TypeError, ValueError, NotImplementedError):
            model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), **kwargs)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), **kwargs)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def predict_to_csv_v10(
    *,
    items_path: Path,
    matches_path: Path,
    model_dir: Path,
    output_path: Path,
    max_length: int = DEFAULT_MAX_LENGTH,
    max_chars: int = DEFAULT_MAX_CHARS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    started = time.perf_counter()
    previous = started

    pairs = pd.read_parquet(matches_path, columns=["id1", "id2"])
    needed = pd.unique(pd.concat([pairs["id1"], pairs["id2"]], ignore_index=True))
    previous = _phase("read-pairs", started, previous, pairs=len(pairs), items=len(needed))

    workers = serialization_workers(len(needed))
    texts, _categories = build_v7_text_cache_from_parquet(
        Path(items_path), needed, max_chars=int(max_chars), workers=workers
    )
    previous = _phase("serialize-items", started, previous, items=len(texts), workers=workers)

    device = _select_device()
    model, tokenizer = _load_model(Path(model_dir), device)
    previous = _phase("load-model", started, previous, device=device)

    score, inference = predict_pairs(
        model=model,
        tokenizer=tokenizer,
        frame=pairs,
        texts=texts,
        device=device,
        max_length=int(max_length),
        batch_size=int(batch_size),
    )
    previous = _phase(
        "score",
        started,
        previous,
        pairs_per_second=round(float(inference["examples_per_second"]), 2),
        batch_size_final=int(inference["batch_size_final"]),
    )

    if len(score) != len(pairs):
        raise RuntimeError(f"v10 produced {len(score)} scores for {len(pairs)} pairs")
    if not np.isfinite(score).all():
        raise RuntimeError("v10 produced non-finite scores")

    result = pairs[["id1", "id2"]].copy()
    result["predict"] = np.clip(score, 0.0, 1.0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    _phase("write", started, previous, rows=len(result), output=str(output_path))
    return result


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MAX_LENGTH",
    "EXPECTED_BASE_MODEL",
    "predict_to_csv_v10",
    "validate_v10_metadata",
]
