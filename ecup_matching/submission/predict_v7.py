"""v7 submission inference: identity-first cross-encoder, no structured phase.

The v7 fold-0 diagnostic scored the cross-encoder on its own, so the submission
runs exactly that: serialize each item once with the identity-first serializer,
then score every pair with one cross-encoder forward pass. There is no TF-IDF
specialist, no explicit-attribute model and no meta fusion, which removes the
entire pure-Python structured phase that made v5/v6 miss the time limit.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

# Imported from v7_runtime, not v7_neural: the training module pulls in the
# split/metric/validation graph, which must not be packaged into the offline
# archive. Both halves share these functions, so they cannot drift.
from ecup_matching.ml.v7_runtime import build_v7_text_cache_from_parquet, predict_pairs


DEFAULT_MAX_LENGTH = 256
DEFAULT_MAX_CHARS = 900
# Measured best on ecup-rtx2060 (run 31547513168: 400.44 pairs/s at batch 64,
# versus 394.70 at batch 256). Larger is not assumed better.
DEFAULT_BATCH_SIZE = 64


def _phase(label: str, started: float, previous: float, **fields: Any) -> float:
    now = time.perf_counter()
    payload = {
        "phase": label,
        "seconds": round(now - previous, 3),
        "total_seconds": round(now - started, 3),
        **fields,
    }
    print(f"[v7] {json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)
    return now


def validate_v7_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Reject packaging mistakes that would misreport what this archive is.

    The fold-0 probe number is a single-fold diagnostic of a model trained
    without that fold. It is not a validated out-of-fold score and must never be
    written into ``strict_oof_macro_average_precision``.
    """
    payload = dict(metadata)

    diagnostic = payload.get("diagnostic_fold0_macro_average_precision")
    if diagnostic is not None and not (0.0 <= float(diagnostic) <= 1.0):
        raise ValueError("diagnostic_fold0_macro_average_precision must be in [0,1]")

    strict = payload.get("strict_oof_macro_average_precision")
    if strict is not None:
        if diagnostic is not None and float(strict) == float(diagnostic):
            raise ValueError(
                "strict_oof_macro_average_precision must not repeat the fold-0 "
                "diagnostic value; a single held fold is not an out-of-fold score"
            )
        if not (0.0 <= float(strict) <= 1.0):
            raise ValueError("strict_oof_macro_average_precision must be in [0,1]")

    if payload.get("gold_metric_opened") is not False or int(
        payload.get("gold_rows_scored", -1)
    ) != 0:
        raise ValueError("packaged v7 metadata violates the sealed gold contract")
    return payload


def _select_device() -> str:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is present in the organizer image
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


def predict_to_csv_v7(
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

    texts, _categories = build_v7_text_cache_from_parquet(
        Path(items_path), needed, max_chars=int(max_chars)
    )
    previous = _phase("serialize-items", started, previous, items=len(texts))

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
        raise RuntimeError(
            f"v7 produced {len(score)} scores for {len(pairs)} pairs"
        )
    if not np.isfinite(score).all():
        raise RuntimeError("v7 produced non-finite scores")

    result = pairs[["id1", "id2"]].copy()
    result["predict"] = np.clip(score, 0.0, 1.0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    _phase("write", started, previous, rows=len(result), output=str(output_path))
    print(
        f"[v7] done rows={len(result):,} total_seconds={time.perf_counter()-started:.2f}",
        flush=True,
    )
    return result
