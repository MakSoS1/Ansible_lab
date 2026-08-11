from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .reranker_data import build_reranker_examples
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_contrastive_data import select_fold_contrastive_pairs
from .v5_evaluation import macro_ap_report
from .v5_validation import manifest_sha256


def teacher_trainable_layer_indices(layer_count: int, *, last_n: int) -> list[int]:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if last_n <= 0 or last_n > layer_count:
        raise ValueError("last_n must be between 1 and layer_count")
    return list(range(layer_count - last_n, layer_count))


def teacher_accumulation_should_step(
    microbatch_count: int,
    *,
    accumulation_steps: int,
    is_last_microbatch: bool,
) -> bool:
    if microbatch_count <= 0:
        raise ValueError("microbatch_count must be positive")
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    return bool(microbatch_count % accumulation_steps == 0 or is_last_microbatch)


def _find_bert_layers(model) -> list[Any]:
    candidates = [
        getattr(getattr(getattr(model, "bert", None), "encoder", None), "layer", None),
        getattr(getattr(getattr(model, "roberta", None), "encoder", None), "layer", None),
        getattr(getattr(model, "encoder", None), "layer", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return list(candidate)
    raise RuntimeError("unable to locate encoder layers")


def train_teacher_fold(
    *,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    base_oof_path: Path,
    model_dir: Path,
    output_dir: Path,
    expected_split_sha: str,
    held_fold: int,
    device: str = "mps",
    physical_batch_size: int = 8,
    effective_batch_size: int = 32,
    predict_batch_size: int = 48,
    max_length: int = 128,
    last_n_layers: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_steps: int = 600,
    max_negative_to_positive: float = 2.0,
    hard_negative_fraction: float = 0.5,
    seed: int = 2026,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if physical_batch_size <= 0 or effective_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if effective_batch_size < physical_batch_size:
        raise ValueError("effective batch must be >= physical batch")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    accumulation_steps = int(np.ceil(effective_batch_size / physical_batch_size))
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed + held_fold)
    np.random.seed(seed + held_fold)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = manifest_sha256(manifest)
    if actual_sha != expected_split_sha:
        raise ValueError(f"sealed split SHA mismatch: {actual_sha}")
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(matches))
    if held_fold not in set(fold_ids.tolist()):
        raise ValueError(f"held fold {held_fold} not present")
    dev_pairs = matches.iloc[dev_rows].reset_index(drop=True)

    wanted_ids = pd.unique(pd.concat([dev_pairs["id1"], dev_pairs["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=True)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev_pairs["category"] = dev_pairs["id1"].map(category_by_id)
    if dev_pairs["category"].isna().any():
        raise RuntimeError("failed to attach development categories")

    gold_rows = np.asarray(manifest["gold_rows"], dtype=np.int64)
    gold_pairs = matches.iloc[gold_rows]
    gold_ids = set(gold_pairs["id1"].tolist()) | set(gold_pairs["id2"].tolist())
    dev_ids = set(items["id"].tolist())
    if gold_ids & dev_ids:
        raise RuntimeError("gold items leaked into teacher item set")

    base_oof = pd.read_parquet(base_oof_path, columns=["row_index", "score"]).sort_values("row_index")
    if base_oof["row_index"].astype(np.int64).tolist() != dev_rows.tolist():
        raise ValueError("base OOF rows do not align with sealed development rows")
    base_scores = base_oof["score"].to_numpy(dtype=np.float64)

    curriculum = select_fold_contrastive_pairs(
        dev_pairs,
        fold_ids,
        base_scores,
        held_fold=held_fold,
        max_negative_to_positive=max_negative_to_positive,
        hard_negative_fraction=hard_negative_fraction,
        seed=seed + 100,
    )
    held_mask = fold_ids == held_fold
    held_pairs = dev_pairs.loc[held_mask].reset_index(drop=True)
    held_rows = dev_rows[held_mask]

    train_frame = build_reranker_examples(items, curriculum)
    valid_frame = build_reranker_examples(items, held_pairs)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
        num_labels=1,
        ignore_mismatched_sizes=True,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = _find_bert_layers(model)
    selected_layers = teacher_trainable_layer_indices(len(layers), last_n=last_n_layers)
    for layer_idx in selected_layers:
        for parameter in layers[layer_idx].parameters():
            parameter.requires_grad = True
    for name, parameter in model.named_parameters():
        if name.startswith("classifier") or ".classifier." in name or "pooler" in name:
            parameter.requires_grad = True
        if name.endswith("LayerNorm.weight") or name.endswith("LayerNorm.bias"):
            parameter.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("teacher has no trainable parameters")

    class PairDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.frame = frame.reset_index(drop=True)
        def __len__(self):
            return len(self.frame)
        def __getitem__(self, idx):
            row = self.frame.iloc[idx]
            return row.text_a, row.text_b, float(row.target)

    def collate(batch):
        a, b, target = zip(*batch)
        tokens = tokenizer(
            list(a), list(b), padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        return tokens, torch.tensor(target, dtype=torch.float32)

    generator = torch.Generator().manual_seed(seed + held_fold)
    loader = DataLoader(
        PairDataset(train_frame),
        batch_size=physical_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    optimizer_steps = 0
    while optimizer_steps < max_steps:
        accumulated = 0
        for batch_idx, (tokens, target) in enumerate(loader):
            tokens = {k: v.to(device) for k, v in tokens.items()}
            target = target.to(device)
            logits = model(**tokens).logits.squeeze(-1)
            raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            (raw_loss / float(accumulation_steps)).backward()
            losses.append(float(raw_loss.detach().cpu()))
            accumulated += 1
            is_last = batch_idx == len(loader) - 1
            if teacher_accumulation_should_step(
                accumulated, accumulation_steps=accumulation_steps, is_last_microbatch=is_last
            ):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accumulated = 0
                if optimizer_steps == 1 or optimizer_steps % 100 == 0:
                    print(
                        f"teacher fold={held_fold} step={optimizer_steps}/{max_steps} "
                        f"loss={losses[-1]:.6f}", flush=True
                    )
                if optimizer_steps >= max_steps:
                    break
        if len(loader) == 0:
            raise RuntimeError("empty teacher dataloader")

    model.eval()
    predict_loader = DataLoader(
        PairDataset(valid_frame), batch_size=predict_batch_size, shuffle=False,
        num_workers=0, collate_fn=collate,
    )
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for tokens, _target in predict_loader:
            tokens = {k: v.to(device) for k, v in tokens.items()}
            logits = model(**tokens).logits.squeeze(-1)
            predictions.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
    score = np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)
    report = macro_ap_report(held_pairs, score)

    pd.DataFrame(
        {
            "row_index": held_rows,
            "fold": np.full(len(held_rows), held_fold, dtype=np.int8),
            "teacher_score": score,
        }
    ).sort_values("row_index").to_parquet(
        output_dir / f"v5f-teacher-fold-{held_fold}-oof.parquet", index=False
    )
    payload = {
        "version": "v5f-rubert-teacher-sprint",
        "held_fold": int(held_fold),
        "split_sha256": expected_split_sha,
        "gold_metric_opened": False,
        "gold_rows_used": 0,
        "gold_items_used": 0,
        "train_rows_selected": int(len(train_frame)),
        "train_positives": int((train_frame["target"] >= 0.5).sum()),
        "train_negatives": int((train_frame["target"] < 0.5).sum()),
        "held_rows": int(len(valid_frame)),
        "steps": int(optimizer_steps),
        "physical_batch_size": int(physical_batch_size),
        "effective_batch_size": int(effective_batch_size),
        "gradient_accumulation_steps": int(accumulation_steps),
        "last_n_layers": int(last_n_layers),
        "max_length": int(max_length),
        "mean_training_loss": float(np.mean(losses)),
        "final_training_loss": float(losses[-1]),
        "held_teacher_macro_ap": float(report["macro_average_precision"]),
        "held_per_category_ap": report["per_category_ap"],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output_dir / f"v5f-teacher-fold-{held_fold}-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-oof", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    parser.add_argument("--held-fold", type=int, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--physical-batch-size", type=int, default=8)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--predict-batch-size", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--last-n-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()
    payload = train_teacher_fold(
        items_path=args.items, matches_path=args.matches, manifest_path=args.manifest,
        base_oof_path=args.base_oof, model_dir=args.model_dir, output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha, held_fold=args.held_fold,
        device=args.device, physical_batch_size=args.physical_batch_size,
        effective_batch_size=args.effective_batch_size, predict_batch_size=args.predict_batch_size,
        max_length=args.max_length, last_n_layers=args.last_n_layers,
        learning_rate=args.learning_rate, max_steps=args.max_steps,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
