from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .train_v1 import attach_pair_category
from .train_v2_structured import prefilter_weak_candidates_parquet
from .train_v5_teacher2_outer_fold import CategoryBatchSampler
from .train_v5_teacher_fold import _find_bert_layers, teacher_accumulation_should_step, teacher_trainable_layer_indices
from .v5_production import select_full_contrastive_pairs
from .v5_teacher2_objective import source_loss_weights, torch_category_ranking_loss
from .v5_validation import manifest_sha256
from .weak_labels import prepare_weak_pairs, remove_human_conflicts, sample_weak_training


def _legacy_text_modules(legacy_package_dir: Path):
    package_dir = legacy_package_dir.resolve()
    if package_dir.name != "legacy_ecup":
        raise ValueError("legacy package directory must be named legacy_ecup")
    sys.path.insert(0, str(package_dir.parent))
    return (
        importlib.import_module("legacy_ecup.ml.textnorm"),
        importlib.import_module("legacy_ecup.ml.v5_item_text"),
    )


def train_production_teacher(
    *,
    human_items_path: Path,
    full_items_path: Path,
    matches_path: Path,
    weak_matches_path: Path,
    manifest_path: Path,
    base_oof_path: Path,
    legacy_package_dir: Path,
    model_dir: Path,
    output_dir: Path,
    expected_split_sha: str,
    device: str = "mps",
    weak_presample_rows: int = 180_000,
    weak_final_rows: int = 100_000,
    physical_batch_size: int = 8,
    effective_batch_size: int = 32,
    max_length: int = 128,
    last_n_layers: int = 4,
    max_steps: int = 800,
    learning_rate: float = 2e-5,
    ranking_weight: float = 0.25,
    seed: int = 2026,
) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_textnorm, legacy_item_text = _legacy_text_modules(legacy_package_dir)
    torch.manual_seed(seed)
    np.random.seed(seed)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_sha256(manifest) != expected_split_sha:
        raise ValueError("sealed split SHA mismatch")
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, _ = development_rows_and_folds(manifest, total_rows=len(matches))
    human_items = pd.read_parquet(
        human_items_path, columns=["id", "name", "attributes", "category"]
    )
    dev = attach_pair_category(matches.iloc[dev_rows].reset_index(drop=True), human_items)
    base = pd.read_parquet(base_oof_path, columns=["row_index", "score"]).sort_values("row_index")
    if base["row_index"].astype(np.int64).tolist() != dev_rows.tolist():
        raise ValueError("base OOF mismatch")
    base_scores = base["score"].to_numpy(dtype=np.float64)

    human_train = select_full_contrastive_pairs(
        dev,
        base_scores,
        max_negative_to_positive=2.0,
        hard_negative_fraction=0.5,
        seed=seed,
    )[["id1", "id2", "target", "category"]].copy()
    human_train["source"] = "human"
    human_train["weak_weight"] = 1.0

    gold_rows = np.asarray(manifest.get("gold_rows", []), dtype=np.int64)
    if len(gold_rows):
        if gold_rows.min() < 0 or gold_rows.max() >= len(matches):
            raise IndexError("manifest contains out-of-range gold row")
        gold = matches.iloc[gold_rows]
        forbidden = set(gold["id1"].tolist()) | set(gold["id2"].tolist())
    else:
        forbidden = set()

    weak, weak_input = prefilter_weak_candidates_parquet(
        weak_matches_path,
        validation_item_ids=forbidden,
        max_presample_rows=weak_presample_rows,
        seed=seed,
    )
    weak, prep = prepare_weak_pairs(weak[["id1", "id2", "target"]])
    weak, conflicts = remove_human_conflicts(
        weak, human_train[["id1", "id2", "target"]]
    )
    weak_ids = set(weak["id1"].tolist()) | set(weak["id2"].tolist())
    weak_items = select_items_by_ids(full_items_path, weak_ids, include_attributes=True)
    weak = attach_pair_category(weak, weak_items)
    weak = sample_weak_training(weak, max_rows=weak_final_rows, seed=seed)
    final_weak_ids = set(weak["id1"].tolist()) | set(weak["id2"].tolist())
    if final_weak_ids & forbidden:
        raise RuntimeError("sealed-gold item leaked into production teacher weak curriculum")
    weak = weak[["id1", "id2", "target", "category", "weak_weight"]].copy()
    weak["source"] = "weak"
    pairs = pd.concat([human_train, weak], ignore_index=True)

    needed = set(pairs["id1"].tolist()) | set(pairs["id2"].tolist())
    human_subset = human_items[human_items["id"].isin(needed)].copy()
    missing = needed - set(human_subset["id"].tolist())
    extra = (
        select_items_by_ids(full_items_path, missing, include_attributes=True)
        if missing
        else human_subset.iloc[:0].copy()
    )
    items = pd.concat([human_subset, extra], ignore_index=True).drop_duplicates("id", keep="first")
    texts: dict[object, str] = {}
    for item_id, name, attributes, category in items[
        ["id", "name", "attributes", "category"]
    ].itertuples(index=False, name=None):
        norm = legacy_textnorm.normalize_item(item_id, name, attributes, category)
        texts[item_id] = (
            f"[CAT] {norm.category}\n"
            f"{legacy_item_text.serialize_item_v5(norm, max_chars=850)}"
        )
    if not needed <= set(texts):
        raise RuntimeError("missing production teacher text")

    categories = sorted(pairs["category"].astype(str).unique().tolist())
    cat_to_id = {cat: i for i, cat in enumerate(categories)}
    pairs["_cat_id"] = pairs["category"].astype(str).map(cat_to_id).astype(int)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True, num_labels=1, ignore_mismatched_sizes=True
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = _find_bert_layers(model)
    for layer_idx in teacher_trainable_layer_indices(len(layers), last_n=last_n_layers):
        for parameter in layers[layer_idx].parameters():
            parameter.requires_grad = True
    for name, parameter in model.named_parameters():
        if (
            name.startswith("classifier")
            or ".classifier." in name
            or "pooler" in name
            or name.endswith("LayerNorm.weight")
            or name.endswith("LayerNorm.bias")
        ):
            parameter.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable teacher parameters selected")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    accumulation_steps = int(np.ceil(effective_batch_size / physical_batch_size))

    class PairDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.frame = frame.reset_index(drop=True)
        def __len__(self):
            return len(self.frame)
        def __getitem__(self, idx):
            row = self.frame.iloc[idx]
            return (
                row["id1"], row["id2"], float(row["target"]), str(row["source"]),
                float(row["weak_weight"]), int(row["_cat_id"]),
            )

    def collate(batch):
        left, right, target, source, confidence, cat = zip(*batch)
        tokens = tokenizer(
            [texts[x] for x in left],
            [texts[x] for x in right],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        weights = source_loss_weights(
            np.asarray(source), np.asarray(confidence), human_weight=1.0, weak_scale=0.1
        )
        return (
            tokens,
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            torch.tensor(cat, dtype=torch.long),
        )

    loader = DataLoader(
        PairDataset(pairs),
        batch_sampler=CategoryBatchSampler(pairs, physical_batch_size, seed),
        num_workers=0,
        collate_fn=collate,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    steps = 0
    losses: list[float] = []
    while steps < max_steps:
        accumulated = 0
        for batch_idx, (tokens, target, weights, cat_ids) in enumerate(loader):
            tokens = {k: v.to(device) for k, v in tokens.items()}
            target = target.to(device)
            weights = weights.to(device)
            cat_ids = cat_ids.to(device)
            logits = model(**tokens).logits.squeeze(-1)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            bce = (bce * weights).sum() / weights.sum().clamp_min(1e-6)
            rank = torch_category_ranking_loss(logits, target, cat_ids)
            loss = bce + ranking_weight * rank
            (loss / accumulation_steps).backward()
            losses.append(float(loss.detach().cpu()))
            accumulated += 1
            if teacher_accumulation_should_step(
                accumulated,
                accumulation_steps=accumulation_steps,
                is_last_microbatch=batch_idx == len(loader) - 1,
            ):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                steps += 1; accumulated = 0
                if steps % 100 == 0 or steps == 1:
                    print(f"step={steps}/{max_steps} loss={losses[-1]:.6f}", flush=True)
                if steps >= max_steps:
                    break

    model.to("cpu")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    payload = {
        "version": "v5-production-field-aware-teacher-v1",
        "split_sha256": expected_split_sha,
        "gold_rows_used": 0,
        "human_rows": int(len(human_train)),
        "weak_rows": int(len(weak)),
        "weak_input_rows": int(weak_input),
        "weak_prepare": prep,
        "weak_conflicts": conflicts,
        "steps": int(steps),
        "max_length": int(max_length),
        "ranking_weight": float(ranking_weight),
        "legacy_source_commit": "cb350b4e7ba6",
        "mean_training_loss": float(np.mean(losses[-100:])) if losses else None,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "production-metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    for name in (
        "human-items", "full-items", "matches", "weak-matches", "manifest", "base-oof",
        "legacy-package-dir", "model-dir", "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-split-sha", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--max-steps", type=int, default=800)
    args = p.parse_args()
    result = train_production_teacher(
        human_items_path=args.human_items,
        full_items_path=args.full_items,
        matches_path=args.matches,
        weak_matches_path=args.weak_matches,
        manifest_path=args.manifest,
        base_oof_path=args.base_oof,
        legacy_package_dir=args.legacy_package_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha,
        device=args.device,
        max_steps=args.max_steps,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
