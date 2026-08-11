from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .textnorm import normalize_item
from .v5_contrastive_data import select_fold_contrastive_pairs
from .v5_embeddings import EMBEDDING_PAIR_FEATURE_NAMES, build_embedding_pair_features
from .v5_evaluation import macro_ap_report
from .v5_item_text import serialize_item_v5
from .v5_validation import manifest_sha256


def contrastive_loss_numpy(cosine, target, *, negative_margin: float = 0.30) -> float:
    score = np.asarray(cosine, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if score.shape != y.shape:
        raise ValueError("cosine and target shapes differ")
    if not 0.0 <= negative_margin < 1.0:
        raise ValueError("negative_margin must be in [0,1)")
    if ((y < 0) | (y > 1)).any():
        raise ValueError("target must be in [0,1]")
    positive = y * np.square(1.0 - score)
    negative = (1.0 - y) * np.square(np.maximum(score - negative_margin, 0.0))
    return float(np.mean(positive + negative))


def trainable_layer_indices(layer_count: int, *, last_n: int) -> list[int]:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if last_n <= 0 or last_n > layer_count:
        raise ValueError("last_n must be between 1 and layer_count")
    return list(range(layer_count - last_n, layer_count))


def _find_encoder_layers(model) -> list[Any]:
    candidates = [
        getattr(getattr(model, "encoder", None), "layer", None),
        getattr(getattr(model, "transformer", None), "layer", None),
        getattr(getattr(getattr(model, "bert", None), "encoder", None), "layer", None),
        getattr(getattr(getattr(model, "roberta", None), "encoder", None), "layer", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return list(candidate)
    raise RuntimeError("unable to locate transformer encoder layers")


def _mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    embedding = summed / denom
    return torch.nn.functional.normalize(embedding, p=2, dim=1)


def _text_cache(items: pd.DataFrame, *, max_chars: int = 700) -> dict[object, str]:
    result: dict[object, str] = {}
    for item_id, name, attributes, category in items[["id", "name", "attributes", "category"]].itertuples(index=False, name=None):
        norm = normalize_item(item_id, name, attributes, category)
        result[item_id] = serialize_item_v5(norm, max_chars=max_chars)
    return result


def _encode_texts(model, tokenizer, texts: list[str], *, device: str, batch_size: int, max_length: int) -> np.ndarray:
    import torch

    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            hidden = model(**tokens).last_hidden_state
            embedding = _mean_pool(hidden, tokens["attention_mask"])
            output.append(embedding.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(output, axis=0) if output else np.empty((0, int(model.config.hidden_size)), dtype=np.float32)


def train_fold(
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
    batch_size: int = 96,
    encode_batch_size: int = 256,
    max_seq_length: int = 96,
    last_n_layers: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    negative_margin: float = 0.30,
    max_steps: int = 800,
    max_negative_to_positive: float = 2.0,
    hard_negative_fraction: float = 0.5,
    seed: int = 2026,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer

    started = time.perf_counter()
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    torch.manual_seed(seed + held_fold)
    np.random.seed(seed + held_fold)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_split_sha = manifest_sha256(manifest)
    if actual_split_sha != expected_split_sha:
        raise ValueError(f"sealed split SHA mismatch: {actual_split_sha}")
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(matches))
    if held_fold not in set(fold_ids.tolist()):
        raise ValueError(f"held fold {held_fold} not present")
    dev_pairs = matches.iloc[dev_rows].reset_index(drop=True)

    all_dev_ids = pd.unique(pd.concat([dev_pairs["id1"], dev_pairs["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, all_dev_ids, include_attributes=True)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev_pairs["category"] = dev_pairs["id1"].map(category_by_id)
    if dev_pairs["category"].isna().any():
        raise RuntimeError("failed to attach pair category")

    gold_rows = np.asarray(manifest["gold_rows"], dtype=np.int64)
    gold_pairs = matches.iloc[gold_rows]
    gold_ids = set(gold_pairs["id1"].tolist()) | set(gold_pairs["id2"].tolist())
    dev_ids = set(items["id"].tolist())
    if gold_ids & dev_ids:
        raise RuntimeError("gold items leaked into contrastive item set")

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
        seed=seed,
    )
    held_mask = fold_ids == held_fold
    held_pairs = dev_pairs.loc[held_mask].reset_index(drop=True)
    held_rows = dev_rows[held_mask]

    texts = _text_cache(items)

    class PairDataset(Dataset):
        def __init__(self, pair_frame: pd.DataFrame):
            self.frame = pair_frame.reset_index(drop=True)

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, index):
            row = self.frame.iloc[index]
            return row.id1, row.id2, float(row.target)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = _find_encoder_layers(model)
    selected_layers = trainable_layer_indices(len(layers), last_n=last_n_layers)
    for layer_index in selected_layers:
        for parameter in layers[layer_index].parameters():
            parameter.requires_grad = True
    # Keep final layer normalization / pooler trainable when present.
    for name, parameter in model.named_parameters():
        if "pooler" in name or name.endswith("LayerNorm.weight") or name.endswith("LayerNorm.bias"):
            parameter.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters selected")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

    def collate(batch):
        left_ids, right_ids, labels = zip(*batch)
        left = tokenizer(
            [texts[x] for x in left_ids], padding=True, truncation=True,
            max_length=max_seq_length, return_tensors="pt",
        )
        right = tokenizer(
            [texts[x] for x in right_ids], padding=True, truncation=True,
            max_length=max_seq_length, return_tensors="pt",
        )
        return left, right, torch.tensor(labels, dtype=torch.float32)

    loader = DataLoader(
        PairDataset(curriculum),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
    )
    model.train()
    losses: list[float] = []
    step = 0
    while step < max_steps:
        for left, right, target in loader:
            left = {k: v.to(device) for k, v in left.items()}
            right = {k: v.to(device) for k, v in right.items()}
            target = target.to(device)
            emb_left = _mean_pool(model(**left).last_hidden_state, left["attention_mask"])
            emb_right = _mean_pool(model(**right).last_hidden_state, right["attention_mask"])
            cosine = torch.nn.functional.cosine_similarity(emb_left, emb_right)
            positive_loss = target * torch.square(1.0 - cosine)
            negative_loss = (1.0 - target) * torch.square(torch.relu(cosine - negative_margin))
            loss = (positive_loss + negative_loss).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            if step % 100 == 0 or step == 1:
                print(f"fold={held_fold} step={step}/{max_steps} loss={losses[-1]:.6f}", flush=True)
            if step >= max_steps:
                break
        if len(loader) == 0:
            raise RuntimeError("empty contrastive dataloader")

    held_item_ids = pd.unique(pd.concat([held_pairs["id1"], held_pairs["id2"]], ignore_index=True))
    held_texts = [texts[item_id] for item_id in held_item_ids]
    held_embeddings = _encode_texts(
        model, tokenizer, held_texts,
        device=device, batch_size=encode_batch_size, max_length=max_seq_length,
    )
    held_index = {item_id: idx for idx, item_id in enumerate(held_item_ids.tolist())}
    left_index = np.fromiter((held_index[x] for x in held_pairs["id1"].tolist()), dtype=np.int64, count=len(held_pairs))
    right_index = np.fromiter((held_index[x] for x in held_pairs["id2"].tolist()), dtype=np.int64, count=len(held_pairs))
    pair_values = build_embedding_pair_features(held_embeddings[left_index], held_embeddings[right_index])
    pair_features = pd.DataFrame(pair_values, columns=EMBEDDING_PAIR_FEATURE_NAMES)
    cosine = pair_features["embedding_cosine"].to_numpy(dtype=np.float64)
    cosine_report = macro_ap_report(held_pairs, cosine)

    fold_output = pd.DataFrame(
        {
            "row_index": held_rows,
            "fold": np.full(len(held_rows), held_fold, dtype=np.int8),
            **{name: pair_features[name].to_numpy() for name in EMBEDDING_PAIR_FEATURE_NAMES},
        }
    ).sort_values("row_index")
    fold_output.to_parquet(output_dir / f"v5d-fold-{held_fold}-oof.parquet", index=False)
    payload = {
        "version": "v5d-contrastive-human-sprint",
        "held_fold": int(held_fold),
        "split_sha256": expected_split_sha,
        "gold_metric_opened": False,
        "gold_rows_used": 0,
        "gold_items_used": 0,
        "train_rows_selected": int(len(curriculum)),
        "train_positives": int((curriculum["target"] >= 0.5).sum()),
        "train_negatives": int((curriculum["target"] < 0.5).sum()),
        "held_rows": int(len(held_pairs)),
        "held_items": int(len(held_item_ids)),
        "steps": int(step),
        "last_n_layers": int(last_n_layers),
        "negative_margin": float(negative_margin),
        "mean_training_loss": float(np.mean(losses)),
        "final_training_loss": float(losses[-1]),
        "held_cosine_macro_ap": float(cosine_report["macro_average_precision"]),
        "held_per_category_ap": cosine_report["per_category_ap"],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output_dir / f"v5d-fold-{held_fold}-metrics.json").write_text(
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
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=96)
    parser.add_argument("--last-n-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--negative-margin", type=float, default=0.30)
    parser.add_argument("--max-steps", type=int, default=800)
    args = parser.parse_args()
    payload = train_fold(
        items_path=args.items, matches_path=args.matches, manifest_path=args.manifest,
        base_oof_path=args.base_oof, model_dir=args.model_dir, output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha, held_fold=args.held_fold,
        device=args.device, batch_size=args.batch_size, encode_batch_size=args.encode_batch_size,
        max_seq_length=args.max_seq_length, last_n_layers=args.last_n_layers,
        learning_rate=args.learning_rate, negative_margin=args.negative_margin,
        max_steps=args.max_steps,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
