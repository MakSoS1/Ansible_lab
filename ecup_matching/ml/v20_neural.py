from __future__ import annotations

import gc
import math
import time
from contextlib import nullcontext
from typing import Mapping

from .v20_strata import REASON_CODES


REASON_TO_ID = {reason: index for index, reason in enumerate(REASON_CODES)}


def _torch():
    import torch
    return torch


class V20MultiTaskModel:
    """Training-only auxiliary heads around the proven sequence-classification parent.

    This class deliberately does not subclass torch.nn.Module at import time so
    repository metadata/audit utilities can import without eagerly importing
    torch. __new__ constructs the real module lazily.
    """

    def __new__(cls, base_model, *, reason_classes: int = len(REASON_CODES)):
        torch = _torch()

        class _Impl(torch.nn.Module):
            def __init__(self, parent, classes: int):
                super().__init__()
                self.base_model = parent
                hidden = int(parent.config.hidden_size)
                self.model_conflict_head = torch.nn.Linear(hidden, 1)
                self.numeric_conflict_head = torch.nn.Linear(hidden, 1)
                self.variant_conflict_head = torch.nn.Linear(hidden, 1)
                self.accessory_head = torch.nn.Linear(hidden, 1)
                self.reason_head = torch.nn.Linear(hidden, int(classes))

            def forward(self, **tokens):
                result = self.base_model(
                    **tokens,
                    output_hidden_states=True,
                    return_dict=True,
                )
                cls_hidden = result.hidden_states[-1][:, 0, :]
                return {
                    "match_logits": result.logits.squeeze(-1),
                    "model_conflict_logits": self.model_conflict_head(cls_hidden).squeeze(-1),
                    "numeric_conflict_logits": self.numeric_conflict_head(cls_hidden).squeeze(-1),
                    "variant_conflict_logits": self.variant_conflict_head(cls_hidden).squeeze(-1),
                    "accessory_logits": self.accessory_head(cls_hidden).squeeze(-1),
                    "reason_logits": self.reason_head(cls_hidden),
                }

        return _Impl(base_model, reason_classes)


def production_base_model(model):
    return model.base_model if hasattr(model, "base_model") else model


def _weighted_bce(logits, target, weights):
    torch = _torch()
    element = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    w = weights.clamp_min(0.0)
    return (element * w).sum() / w.sum().clamp_min(1e-6)


def compute_v20_loss(
    outputs: Mapping[str, object],
    target,
    match_weight,
    *,
    aux_targets: Mapping[str, object],
    aux_mask,
    lambda_reason: float = 0.15,
    lambda_consistency: float = 0.05,
    swapped_match_logits=None,
) -> dict[str, object]:
    torch = _torch()
    match = _weighted_bce(outputs["match_logits"], target, match_weight)
    aux_total = torch.zeros((), dtype=match.dtype, device=match.device)
    mask = aux_mask.float().clamp(0.0, 1.0)
    mask_sum = mask.sum()
    binary_pairs = [
        ("model_conflict", "model_conflict_logits"),
        ("numeric_conflict", "numeric_conflict_logits"),
        ("variant_conflict", "variant_conflict_logits"),
        ("accessory", "accessory_logits"),
    ]
    if float(mask_sum.detach().cpu()) > 0:
        terms = []
        for target_key, logit_key in binary_pairs:
            if target_key not in aux_targets or logit_key not in outputs:
                continue
            element = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs[logit_key], aux_targets[target_key].float(), reduction="none"
            )
            terms.append((element * mask).sum() / mask_sum.clamp_min(1e-6))
        if "reason" in aux_targets and "reason_logits" in outputs:
            element = torch.nn.functional.cross_entropy(
                outputs["reason_logits"], aux_targets["reason"].long(), reduction="none"
            )
            terms.append((element * mask).sum() / mask_sum.clamp_min(1e-6))
        if terms:
            aux_total = torch.stack(terms).mean()

    consistency = torch.zeros((), dtype=match.dtype, device=match.device)
    if swapped_match_logits is not None and float(lambda_consistency) > 0:
        consistency = torch.nn.functional.mse_loss(
            torch.sigmoid(outputs["match_logits"]),
            torch.sigmoid(swapped_match_logits),
        )
    total = match + float(lambda_reason) * aux_total + float(lambda_consistency) * consistency
    return {
        "total": total,
        "match": match,
        "reason": aux_total,
        "consistency": consistency,
    }


def train_v20_phase(
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
    lambda_reason: float = 0.15,
    lambda_consistency: float = 0.05,
    telemetry_every_steps: int = 100,
):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import get_cosine_schedule_with_warmup

    from .v5_teacher2_objective import torch_category_ranking_loss
    from .v7_neural import MacroPairBatchSampler, TrainPhaseResult, _memory_payload

    if effective_batch_size < physical_batch_size:
        raise ValueError("effective_batch_size must be >= physical_batch_size")
    if not math.isfinite(float(epochs)) or float(epochs) <= 0:
        raise ValueError("epochs must be finite and positive")
    required = {"id1", "id2", "target", "category", "match_weight"}
    if not required.issubset(frame.columns):
        raise ValueError(f"v20 frame missing columns: {sorted(required - set(frame.columns))}")

    accumulation_steps = int(math.ceil(effective_batch_size / physical_batch_size))
    work = frame.reset_index(drop=True).copy()
    categories = sorted(work["category"].astype(str).unique())
    cat_to_id = {category: idx for idx, category in enumerate(categories)}
    work["_cat_id"] = work["category"].astype(str).map(cat_to_id).astype(int)
    for column, default in {
        "aux_model_conflict": 0.0,
        "aux_numeric_conflict": 0.0,
        "aux_variant_conflict": 0.0,
        "aux_accessory": 0.0,
        "aux_mask": 0.0,
        "reason_code": "OTHER",
    }.items():
        if column not in work:
            work[column] = default
    work["_reason_id"] = work["reason_code"].astype(str).map(REASON_TO_ID).fillna(REASON_TO_ID["OTHER"]).astype(int)

    class PairDataset(Dataset):
        def __len__(self):
            return len(work)

        def __getitem__(self, index):
            r = work.iloc[index]
            return (
                r.id1, r.id2, float(r.target), float(r.match_weight), int(r._cat_id),
                float(r.aux_model_conflict), float(r.aux_numeric_conflict),
                float(r.aux_variant_conflict), float(r.aux_accessory),
                float(r.aux_mask), int(r._reason_id),
            )

    def collate(batch):
        fields = list(zip(*batch))
        left_ids, right_ids = fields[0], fields[1]
        forward = tokenizer(
            [texts[x] for x in left_ids], [texts[x] for x in right_ids],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        reverse = None
        if float(lambda_consistency) > 0:
            reverse = tokenizer(
                [texts[x] for x in right_ids], [texts[x] for x in left_ids],
                padding=True, truncation=True, max_length=max_length, return_tensors="pt",
            )
        tensors = [torch.tensor(values, dtype=torch.float32) for values in fields[2:4]]
        cat = torch.tensor(fields[4], dtype=torch.long)
        aux = [torch.tensor(values, dtype=torch.float32) for values in fields[5:10]]
        reason = torch.tensor(fields[10], dtype=torch.long)
        return forward, reverse, tensors[0], tensors[1], cat, aux, reason

    sampler = MacroPairBatchSampler(work, physical_batch_size, seed)
    loader = DataLoader(PairDataset(), batch_sampler=sampler, num_workers=0, pin_memory=str(device).startswith("cuda"), collate_fn=collate)
    microbatch_target = max(1, int(math.ceil(len(loader) * float(epochs))))
    optimizer_steps_target = int(math.ceil(microbatch_target / accumulation_steps))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("v20 model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(round(optimizer_steps_target * 0.05))),
        num_training_steps=max(1, optimizer_steps_target),
    )
    use_cuda = str(device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_cuda else None
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    loss_sum = 0.0
    consumed = optimizer_steps = examples_seen = 0
    epoch = 0
    while consumed < microbatch_target:
        sampler.epoch = epoch
        for forward, reverse, target, weight, cat, aux, reason in loader:
            if consumed >= microbatch_target:
                break
            forward = {k: v.to(device, non_blocking=use_cuda) for k, v in forward.items()}
            reverse = {k: v.to(device, non_blocking=use_cuda) for k, v in reverse.items()} if reverse is not None else None
            target = target.to(device, non_blocking=use_cuda)
            weight = weight.to(device, non_blocking=use_cuda)
            cat = cat.to(device, non_blocking=use_cuda)
            aux = [v.to(device, non_blocking=use_cuda) for v in aux]
            reason = reason.to(device, non_blocking=use_cuda)
            amp = torch.cuda.amp.autocast(dtype=torch.float16) if use_cuda else nullcontext()
            with amp:
                outputs = model(**forward)
                swapped = model(**reverse)["match_logits"] if reverse is not None else None
                losses = compute_v20_loss(
                    outputs, target, weight,
                    aux_targets={
                        "model_conflict": aux[0], "numeric_conflict": aux[1],
                        "variant_conflict": aux[2], "accessory": aux[3], "reason": reason,
                    },
                    aux_mask=aux[4], lambda_reason=lambda_reason,
                    lambda_consistency=lambda_consistency, swapped_match_logits=swapped,
                )
                rank = torch_category_ranking_loss(outputs["match_logits"], target, cat)
                loss = losses["total"] + float(ranking_weight) * rank
                scaled = loss / accumulation_steps
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            consumed += 1
            examples_seen += int(target.numel())
            loss_sum += float(loss.detach().cpu())
            if consumed % accumulation_steps == 0 or consumed == microbatch_target:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if scaler is not None:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                scheduler.step(); optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if telemetry_every_steps and (optimizer_steps == 1 or optimizer_steps % telemetry_every_steps == 0 or optimizer_steps == optimizer_steps_target):
                    elapsed = time.perf_counter() - started
                    print({
                        "phase": phase, "optimizer_step": optimizer_steps,
                        "optimizer_steps_total": optimizer_steps_target,
                        "examples_per_second": round(examples_seen / max(elapsed, 1e-9), 2),
                        **_memory_payload(torch),
                    }, flush=True)
        epoch += 1
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = TrainPhaseResult(
        phase=phase, microbatches=consumed, optimizer_steps=optimizer_steps,
        examples_seen=examples_seen, mean_loss=loss_sum / max(consumed, 1),
        elapsed_seconds=elapsed, examples_per_second=examples_seen / max(elapsed, 1e-9),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if use_cuda else 0,
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()) if use_cuda else 0,
    )
    del optimizer, scheduler, loader
    if scaler is not None:
        del scaler
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    return result


__all__ = [
    "REASON_TO_ID", "V20MultiTaskModel", "production_base_model",
    "compute_v20_loss", "train_v20_phase",
]
