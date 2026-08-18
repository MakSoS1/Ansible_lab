from __future__ import annotations

import gc
import math
import time
from contextlib import nullcontext
from typing import Mapping

from .v5_teacher2_objective import torch_category_ranking_loss
from .v7_neural import MacroPairBatchSampler, TrainPhaseResult, _memory_payload
from .v18_ema import ExponentialMovingAverage
from .v18_views import augment_serialized_view, deterministic_decision


def train_pair_phase_v18(
    *,
    model,
    tokenizer,
    frame,
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
    pair_swap_probability: float = 0.0,
    residual_dropout_probability: float = 0.0,
    numeric_dropout_probability: float = 0.0,
    ema: ExponentialMovingAverage | None = None,
    telemetry_every_steps: int = 100,
) -> TrainPhaseResult:
    """v18 training loop with deterministic train-only views and optional EMA.

    Defaults are intentionally behavior-compatible with v7 except that the code
    is device-general enough for a tiny MPS smoke test. Canonical RTX training
    still uses CUDA.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import get_cosine_schedule_with_warmup

    for name, probability in {
        "pair_swap_probability": pair_swap_probability,
        "residual_dropout_probability": residual_dropout_probability,
        "numeric_dropout_probability": numeric_dropout_probability,
    }.items():
        p = float(probability)
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must be finite and in [0,1]")
    if effective_batch_size < physical_batch_size:
        raise ValueError("effective_batch_size must be >= physical_batch_size")
    if not math.isfinite(float(epochs)) or float(epochs) <= 0:
        raise ValueError("epochs must be finite and positive")

    accumulation_steps = int(math.ceil(effective_batch_size / physical_batch_size))
    work = frame.reset_index(drop=True).copy()
    categories = sorted(work["category"].astype(str).unique())
    cat_to_id = {category: idx for idx, category in enumerate(categories)}
    work["_cat_id"] = work["category"].astype(str).map(cat_to_id).astype(int)
    if weak and "weak_weight" not in work:
        raise ValueError("weak phase requires weak_weight")

    class PairDataset(Dataset):
        def __init__(self) -> None:
            self.epoch = 0

        def __len__(self):
            return len(work)

        def __getitem__(self, index):
            row = work.iloc[index]
            return (
                int(index),
                row["id1"],
                row["id2"],
                float(row["target"]),
                float(row["weak_weight"]) if weak else 1.0,
                int(row["_cat_id"]),
            )

    dataset = PairDataset()

    def collate(batch):
        indices, left_ids, right_ids, target, confidence, cat = zip(*batch)
        left_texts: list[str] = []
        right_texts: list[str] = []
        for index, left_id, right_id in zip(indices, left_ids, right_ids):
            if deterministic_decision(
                seed=seed,
                epoch=dataset.epoch,
                index=int(index),
                stream="pair-swap",
                probability=pair_swap_probability,
            ):
                left_id, right_id = right_id, left_id
            left = texts[left_id]
            right = texts[right_id]
            left = augment_serialized_view(
                left,
                drop_residual=deterministic_decision(
                    seed=seed, epoch=dataset.epoch, index=int(index), stream="left-residual",
                    probability=residual_dropout_probability,
                ),
                drop_numeric=deterministic_decision(
                    seed=seed, epoch=dataset.epoch, index=int(index), stream="left-numeric",
                    probability=numeric_dropout_probability,
                ),
            )
            right = augment_serialized_view(
                right,
                drop_residual=deterministic_decision(
                    seed=seed, epoch=dataset.epoch, index=int(index), stream="right-residual",
                    probability=residual_dropout_probability,
                ),
                drop_numeric=deterministic_decision(
                    seed=seed, epoch=dataset.epoch, index=int(index), stream="right-numeric",
                    probability=numeric_dropout_probability,
                ),
            )
            left_texts.append(left)
            right_texts.append(right)
        tokens = tokenizer(
            left_texts,
            right_texts,
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
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=str(device).startswith("cuda"),
        collate_fn=collate,
    )
    microbatch_target = max(1, int(math.ceil(len(loader) * float(epochs))))
    optimizer_steps_target = int(math.ceil(microbatch_target / accumulation_steps))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("v18 model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(round(optimizer_steps_target * 0.05))),
        num_training_steps=max(1, optimizer_steps_target),
    )
    use_cuda_amp = str(device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_cuda_amp else None

    if use_cuda_amp:
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
        dataset.epoch = epoch
        for tokens, target, confidence, cat_ids in loader:
            if consumed >= microbatch_target:
                break
            tokens = {
                key: value.to(device, non_blocking=use_cuda_amp)
                for key, value in tokens.items()
            }
            target = target.to(device, non_blocking=use_cuda_amp)
            confidence = confidence.to(device, non_blocking=use_cuda_amp)
            cat_ids = cat_ids.to(device, non_blocking=use_cuda_amp)
            amp_context = (
                torch.cuda.amp.autocast(dtype=torch.float16) if use_cuda_amp else nullcontext()
            )
            with amp_context:
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
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            consumed += 1
            examples_seen += int(target.numel())
            loss_sum += float(loss.detach().cpu())
            should_step = consumed % accumulation_steps == 0 or consumed == microbatch_target
            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if ema is not None:
                    ema.update(model)
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

    if use_cuda_amp:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = TrainPhaseResult(
        phase=phase,
        microbatches=int(consumed),
        optimizer_steps=int(optimizer_steps),
        examples_seen=int(examples_seen),
        mean_loss=float(loss_sum / max(consumed, 1)),
        elapsed_seconds=float(elapsed),
        examples_per_second=float(examples_seen / max(elapsed, 1e-9)),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if use_cuda_amp else 0,
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()) if use_cuda_amp else 0,
    )
    del optimizer, scheduler, loader, dataset
    if scaler is not None:
        del scaler
    gc.collect()
    if use_cuda_amp:
        torch.cuda.empty_cache()
    return result


__all__ = ["train_pair_phase_v18"]
