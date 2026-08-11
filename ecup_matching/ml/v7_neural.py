from __future__ import annotations

import gc
import math
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .features import normalize_items
from .textnorm import normalize_item
from .train_v5_teacher_fold import _find_bert_layers
from .v5_teacher2_objective import torch_category_ranking_loss
from .v7_item_text import serialize_item_v7


@dataclass(frozen=True)
class TrainPhaseResult:
    phase: str
    microbatches: int
    optimizer_steps: int
    examples_seen: int
    mean_loss: float
    elapsed_seconds: float
    examples_per_second: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int


class MacroPairBatchSampler:
    """Equalize category exposure and keep positive/negative pairs together.

    Macro AP gives every category equal weight, while the raw training table does
    not. Each epoch therefore emits the same number of batches for every category.
    When a category contains both labels, each batch deliberately contains both
    labels (for batch_size >= 2), which makes the category-local ranking objective
    informative instead of frequently receiving one-class microbatches.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        batch_size: int,
        seed: int,
        *,
        epoch: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        required = {"category", "target"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"macro sampler missing columns: {sorted(missing)}")
        if len(frame) == 0:
            raise ValueError("macro sampler requires non-empty frame")
        self.frame = frame.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.categories = sorted(self.frame["category"].astype(str).unique().tolist())
        if not self.categories:
            raise ValueError("macro sampler requires at least one category")
        total_batches = max(1, math.ceil(len(self.frame) / self.batch_size))
        self.batches_per_category = max(1, math.ceil(total_batches / len(self.categories)))

    def __len__(self) -> int:
        return int(self.batches_per_category * len(self.categories))

    @staticmethod
    def _draw(rng: np.random.Generator, pool: np.ndarray, size: int) -> list[int]:
        if size <= 0:
            return []
        if len(pool) == 0:
            raise ValueError("cannot draw from an empty pool")
        replace = len(pool) < size
        return rng.choice(pool, size=size, replace=replace).astype(int).tolist()

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        batches: list[list[int]] = []
        category_values = self.frame["category"].astype(str)
        targets = pd.to_numeric(self.frame["target"], errors="raise").to_numpy(float)
        for category in self.categories:
            idx = np.flatnonzero(category_values.to_numpy() == category)
            pos = idx[targets[idx] >= 0.5]
            neg = idx[targets[idx] < 0.5]
            for _ in range(self.batches_per_category):
                if len(pos) and len(neg) and self.batch_size >= 2:
                    positive_count = max(1, self.batch_size // 2)
                    negative_count = self.batch_size - positive_count
                    if negative_count == 0:
                        negative_count = 1
                        positive_count = self.batch_size - 1
                    batch = self._draw(rng, pos, positive_count) + self._draw(
                        rng, neg, negative_count
                    )
                else:
                    batch = self._draw(rng, idx, self.batch_size)
                rng.shuffle(batch)
                batches.append(batch)
        rng.shuffle(batches)
        return iter(batches)


def phase_microbatches(*, loader_batches: int, epochs: float) -> int:
    if loader_batches <= 0:
        raise ValueError("loader_batches must be positive")
    if not math.isfinite(float(epochs)) or float(epochs) <= 0:
        raise ValueError("epochs must be finite and positive")
    return max(1, int(math.ceil(loader_batches * float(epochs))))


def build_v7_text_cache(
    items: pd.DataFrame,
    *,
    max_chars: int,
    attribute_importance: Mapping[str, float] | None = None,
) -> dict[object, str]:
    normalized = normalize_items(items)
    return {
        item_id: f"[CAT] {norm.category}\n{serialize_item_v7(norm, max_chars=max_chars, attribute_importance=attribute_importance)}"
        for item_id, norm in normalized.items()
    }


def build_v7_text_cache_from_parquet(
    parquet_path: Path,
    item_ids: Iterable[object],
    *,
    max_chars: int,
    batch_size: int = 131_072,
    attribute_importance: Mapping[str, float] | None = None,
) -> tuple[dict[object, str], dict[object, str]]:
    """Stream selected full item records and serialize them without a giant DataFrame.

    v7 weak pretraining depends on canonical typed attributes. The legacy
    `include_attributes=False` fast path intentionally replaced attributes by `{}`;
    that is correct for name-only consumers but silently defeats the v7 hypothesis.
    This scanner keeps the real attributes while retaining only the final serialized
    strings and category map in memory.
    """
    requested = set(item_ids)
    if not requested:
        return {}, {}
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    parquet = pq.ParquetFile(str(parquet_path))
    columns = ("id", "name", "attributes", "category")
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"items parquet missing columns: {sorted(missing)}")
    id_type = parquet.schema_arrow.field("id").type
    value_set = pa.array(list(requested), type=id_type)
    texts: dict[object, str] = {}
    categories: dict[object, str] = {}
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
        id_column = batch.column(batch.schema.get_field_index("id"))
        selected = batch.filter(pc.is_in(id_column, value_set=value_set))
        if selected.num_rows:
            payload = selected.to_pydict()
            for item_id, name, attributes, category in zip(
                payload["id"],
                payload["name"],
                payload["attributes"],
                payload["category"],
            ):
                if item_id in texts:
                    continue
                norm = normalize_item(item_id, name, attributes, category)
                texts[item_id] = (
                    f"[CAT] {norm.category}\n"
                    f"{serialize_item_v7(norm, max_chars=max_chars, attribute_importance=attribute_importance)}"
                )
                categories[item_id] = str(category)
        del id_column, selected, batch
        if len(texts) == len(requested):
            break
    missing_ids = requested - set(texts)
    if missing_ids:
        first = min(missing_ids, key=lambda value: (type(value).__name__, repr(value)))
        raise KeyError(
            f"items parquet is missing {len(missing_ids)} requested IDs; first={first!r}"
        )
    del value_set
    pa.default_memory_pool().release_unused()
    return texts, categories


def configure_trainable_layers(model, *, last_n_layers: int) -> list:
    if last_n_layers <= 0:
        raise ValueError("last_n_layers must be positive")
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = _find_bert_layers(model)
    if not layers:
        raise ValueError("could not find BERT encoder layers")
    start = max(0, len(layers) - int(last_n_layers))
    for layer in layers[start:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for name, parameter in model.named_parameters():
        lowered = name.casefold()
        if (
            "classifier" in lowered
            or "pooler" in lowered
            or lowered.endswith("layernorm.weight")
            or lowered.endswith("layernorm.bias")
        ):
            parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("v7 model has no trainable parameters")
    return trainable


def _memory_payload(torch) -> dict[str, int]:
    rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_bytes = rss_kib * (1024 if os.name != "darwin" else 1)
    payload = {"peak_rss_bytes": int(rss_bytes)}
    if torch.cuda.is_available():
        payload.update(
            cuda_allocated_bytes=int(torch.cuda.memory_allocated()),
            cuda_reserved_bytes=int(torch.cuda.memory_reserved()),
            cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
    return payload


def train_pair_phase(
    *,
    model,
    tokenizer,
    frame: pd.DataFrame,
    texts: Mapping[object, str],
    device: str,
    phase: str,
    epochs: float,
    physical_batch_size: int,
    effective_batch_size: int,
    max_length: int,
    learning_rate: float,
    ranking_weight: float,
    seed: int,
    weak: bool,
    telemetry_every_steps: int = 100,
) -> TrainPhaseResult:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import get_cosine_schedule_with_warmup

    if effective_batch_size < physical_batch_size:
        raise ValueError("effective_batch_size must be >= physical_batch_size")
    accumulation_steps = int(math.ceil(effective_batch_size / physical_batch_size))
    work = frame.reset_index(drop=True).copy()
    categories = sorted(work["category"].astype(str).unique())
    cat_to_id = {category: idx for idx, category in enumerate(categories)}
    work["_cat_id"] = work["category"].astype(str).map(cat_to_id).astype(int)
    if weak and "weak_weight" not in work:
        raise ValueError("weak phase requires weak_weight")

    class PairDataset(Dataset):
        def __len__(self):
            return len(work)

        def __getitem__(self, index):
            row = work.iloc[index]
            return (
                row["id1"],
                row["id2"],
                float(row["target"]),
                float(row["weak_weight"]) if weak else 1.0,
                int(row["_cat_id"]),
            )

    def collate(batch):
        left, right, target, confidence, cat = zip(*batch)
        tokens = tokenizer(
            [texts[item_id] for item_id in left],
            [texts[item_id] for item_id in right],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return (
            tokens,
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(confidence, dtype=torch.float32),
            torch.tensor(cat, dtype=torch.long),
        )

    sampler = MacroPairBatchSampler(work, physical_batch_size, seed)
    loader = DataLoader(
        PairDataset(),
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate,
    )
    microbatch_target = phase_microbatches(loader_batches=len(loader), epochs=epochs)
    optimizer_steps_target = int(math.ceil(microbatch_target / accumulation_steps))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(round(optimizer_steps_target * 0.05))),
        num_training_steps=max(1, optimizer_steps_target),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    loss_sum = 0.0
    optimizer_steps = 0
    examples_seen = 0
    consumed = 0
    epoch = 0
    while consumed < microbatch_target:
        sampler.epoch = epoch
        for tokens, target, confidence, cat_ids in loader:
            if consumed >= microbatch_target:
                break
            tokens = {
                key: value.to(device, non_blocking=device.startswith("cuda"))
                for key, value in tokens.items()
            }
            target = target.to(device, non_blocking=device.startswith("cuda"))
            confidence = confidence.to(device, non_blocking=device.startswith("cuda"))
            cat_ids = cat_ids.to(device, non_blocking=device.startswith("cuda"))
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda"), dtype=torch.float16):
                logits = model(**tokens).logits.squeeze(-1)
                element = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
                if weak:
                    weights = confidence.clamp(0.05, 1.0)
                    bce = (element * weights).sum() / weights.sum().clamp_min(1e-6)
                else:
                    bce = element.mean()
                rank = torch_category_ranking_loss(logits, target, cat_ids)
                loss = bce + float(ranking_weight) * rank
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            consumed += 1
            examples_seen += int(target.numel())
            loss_sum += float(loss.detach().cpu())
            should_step = consumed % accumulation_steps == 0 or consumed == microbatch_target
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if telemetry_every_steps > 0 and (
                    optimizer_steps == 1
                    or optimizer_steps % telemetry_every_steps == 0
                    or optimizer_steps == optimizer_steps_target
                ):
                    elapsed = time.perf_counter() - started
                    rate = examples_seen / max(elapsed, 1e-9)
                    remaining = max(0, optimizer_steps_target - optimizer_steps)
                    eta = remaining * elapsed / max(optimizer_steps, 1)
                    print(
                        {
                            "phase": phase,
                            "optimizer_step": optimizer_steps,
                            "optimizer_steps_total": optimizer_steps_target,
                            "percent": round(100.0 * optimizer_steps / optimizer_steps_target, 2),
                            "elapsed_seconds": round(elapsed, 2),
                            "examples_per_second": round(rate, 2),
                            "eta_seconds": round(eta, 2),
                            **_memory_payload(torch),
                        },
                        flush=True,
                    )
        epoch += 1

    elapsed = time.perf_counter() - started
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    result = TrainPhaseResult(
        phase=phase,
        microbatches=int(consumed),
        optimizer_steps=int(optimizer_steps),
        examples_seen=int(examples_seen),
        mean_loss=float(loss_sum / max(consumed, 1)),
        elapsed_seconds=float(elapsed),
        examples_per_second=float(examples_seen / max(elapsed, 1e-9)),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0,
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()) if device.startswith("cuda") else 0,
    )
    del optimizer, scheduler, scaler, loader
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def predict_pairs(
    *,
    model,
    tokenizer,
    frame: pd.DataFrame,
    texts: Mapping[object, str],
    device: str,
    max_length: int,
    batch_size: int = 16,
) -> tuple[np.ndarray, dict[str, float | int]]:
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    predictions: list[np.ndarray] = []
    started = time.perf_counter()
    current_batch = int(batch_size)
    row = 0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        while row < len(frame):
            stop = min(len(frame), row + current_batch)
            chunk = frame.iloc[row:stop]
            try:
                tokens = tokenizer(
                    [texts[item_id] for item_id in chunk["id1"]],
                    [texts[item_id] for item_id in chunk["id2"]],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokens = {
                    key: value.to(device, non_blocking=device.startswith("cuda"))
                    for key, value in tokens.items()
                }
                with torch.cuda.amp.autocast(
                    enabled=device.startswith("cuda"), dtype=torch.float16
                ):
                    logits = model(**tokens).logits.squeeze(-1)
                predictions.append(torch.sigmoid(logits).float().cpu().numpy())
                row = stop
            except torch.cuda.OutOfMemoryError:
                if not device.startswith("cuda") or current_batch <= 1:
                    raise
                current_batch = max(1, current_batch // 2)
                torch.cuda.empty_cache()
                print(
                    {"phase": "predict", "cuda_oom_batch_halved_to": current_batch},
                    flush=True,
                )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    score = np.concatenate(predictions).astype(np.float64, copy=False) if predictions else np.empty(0)
    return score, {
        "rows": int(len(frame)),
        "seconds": float(elapsed),
        "examples_per_second": float(len(frame) / max(elapsed, 1e-9)),
        "batch_size_final": int(current_batch),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device.startswith("cuda") else 0,
    }