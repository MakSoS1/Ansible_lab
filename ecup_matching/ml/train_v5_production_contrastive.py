from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .train_v5_biencoder_fold import (
    _find_encoder_layers,
    _mean_pool,
    accumulation_should_step,
    trainable_layer_indices,
)
from .v5_production import select_full_contrastive_pairs
from .v5_validation import manifest_sha256


def _legacy_text_modules(legacy_package_dir: Path):
    package_dir = legacy_package_dir.resolve()
    sys.path.insert(0, str(package_dir.parent))
    return (
        importlib.import_module("legacy_ecup.ml.textnorm"),
        importlib.import_module("legacy_ecup.ml.v5_item_text"),
    )


def train_production_contrastive(
    *,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    base_oof_path: Path,
    legacy_package_dir: Path,
    model_dir: Path,
    output_dir: Path,
    expected_split_sha: str,
    device: str = "mps",
    batch_size: int = 96,
    micro_batch_size: int | None = None,
    max_seq_length: int = 96,
    last_n_layers: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    negative_margin: float = 0.30,
    max_steps: int = 800,
    seed: int = 2026,
) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer

    started = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    legacy_textnorm, legacy_item_text = _legacy_text_modules(legacy_package_dir)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_sha256(manifest) != expected_split_sha:
        raise ValueError("sealed split SHA mismatch")
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, _ = development_rows_and_folds(manifest, total_rows=len(matches))
    dev = matches.iloc[dev_rows].reset_index(drop=True)
    ids = pd.unique(pd.concat([dev["id1"], dev["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, ids, include_attributes=True)
    categories = items.set_index("id")["category"].astype(str)
    dev["category"] = dev["id1"].map(categories)
    if dev["category"].isna().any():
        raise RuntimeError("failed to attach development category")

    gold_rows = np.asarray(manifest.get("gold_rows", []), dtype=np.int64)
    if len(gold_rows):
        gold = matches.iloc[gold_rows]
        gold_ids = set(gold["id1"].tolist()) | set(gold["id2"].tolist())
        if gold_ids & set(ids.tolist()):
            raise RuntimeError("gold items leaked into development item set")

    base_oof = pd.read_parquet(base_oof_path, columns=["row_index", "score"]).sort_values("row_index")
    if base_oof["row_index"].astype(np.int64).tolist() != dev_rows.tolist():
        raise ValueError("base OOF rows do not align with development rows")
    curriculum = select_full_contrastive_pairs(dev, base_oof["score"].to_numpy(dtype=np.float64), seed=seed)

    texts: dict[object, str] = {}
    for item_id, name, attributes, category in items[["id", "name", "attributes", "category"]].itertuples(index=False, name=None):
        norm = legacy_textnorm.normalize_item(item_id, name, attributes, category)
        texts[item_id] = legacy_item_text.serialize_item_v5(norm, max_chars=700)

    class PairDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.frame = frame.reset_index(drop=True)
        def __len__(self):
            return len(self.frame)
        def __getitem__(self, index):
            row = self.frame.iloc[index]
            return row.id1, row.id2, float(row.target)

    physical_batch = min(batch_size, 24) if micro_batch_size is None and str(device).startswith("mps") else (micro_batch_size or batch_size)
    physical_batch = int(min(batch_size, physical_batch))
    accumulation_steps = max(1, int(math.ceil(batch_size / physical_batch)))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = _find_encoder_layers(model)
    selected_layers = trainable_layer_indices(len(layers), last_n=last_n_layers)
    for layer_index in selected_layers:
        for parameter in layers[layer_index].parameters():
            parameter.requires_grad = True
    for name, parameter in model.named_parameters():
        if "pooler" in name or name.endswith("LayerNorm.weight") or name.endswith("LayerNorm.bias"):
            parameter.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

    def collate(batch):
        left_ids, right_ids, labels = zip(*batch)
        left = tokenizer([texts[x] for x in left_ids], padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt")
        right = tokenizer([texts[x] for x in right_ids], padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt")
        return left, right, torch.tensor(labels, dtype=torch.float32)

    loader = DataLoader(PairDataset(curriculum), batch_size=physical_batch, shuffle=True, num_workers=0, collate_fn=collate, drop_last=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    micro_count = 0
    losses: list[float] = []
    while optimizer_steps < max_steps:
        for batch_index, (left, right, target) in enumerate(loader):
            left = {k: v.to(device) for k, v in left.items()}
            right = {k: v.to(device) for k, v in right.items()}
            target = target.to(device)
            emb_left = _mean_pool(model(**left).last_hidden_state, left["attention_mask"])
            emb_right = _mean_pool(model(**right).last_hidden_state, right["attention_mask"])
            cosine = torch.nn.functional.cosine_similarity(emb_left, emb_right)
            raw_loss = (
                target * torch.square(1.0 - cosine)
                + (1.0 - target) * torch.square(torch.relu(cosine - negative_margin))
            ).mean()
            (raw_loss / float(accumulation_steps)).backward()
            losses.append(float(raw_loss.detach().cpu()))
            micro_count += 1
            if accumulation_should_step(micro_count, accumulation_steps=accumulation_steps, is_last_microbatch=batch_index == len(loader) - 1):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1; micro_count = 0
                if optimizer_steps % 100 == 0 or optimizer_steps == 1:
                    print(f"step={optimizer_steps}/{max_steps} loss={losses[-1]:.6f}", flush=True)
                if optimizer_steps >= max_steps:
                    break

    output_dir.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    payload = {
        "version": "v5-production-contrastive-v1",
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev)),
        "curriculum_rows": int(len(curriculum)),
        "optimizer_steps": int(optimizer_steps),
        "max_seq_length": int(max_seq_length),
        "gold_rows_used": 0,
        "legacy_source_commit": "cb350b4e7ba6",
        "mean_training_loss": float(np.mean(losses[-100:])) if losses else None,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "production-metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--matches", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--base-oof", type=Path, required=True)
    p.add_argument("--legacy-package-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-split-sha", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--max-steps", type=int, default=800)
    args = p.parse_args()
    result = train_production_contrastive(
        items_path=args.items,
        matches_path=args.matches,
        manifest_path=args.manifest,
        base_oof_path=args.base_oof,
        legacy_package_dir=args.legacy_package_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha,
        device=args.device,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
