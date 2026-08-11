from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .features import normalize_items
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .train_v1 import attach_pair_category
from .train_v2_structured import prefilter_weak_candidates_parquet
from .train_v5_teacher_fold import _find_bert_layers, teacher_accumulation_should_step, teacher_trainable_layer_indices
from .v5_contrastive_data import select_fold_contrastive_pairs
from .v5_evaluation import macro_ap_report
from .v5_item_text import serialize_item_v5
from .v5_teacher2_objective import source_loss_weights, torch_category_ranking_loss
from .v5_validation import manifest_sha256
from .v5_weak_specialists import forbidden_weak_item_ids
from .weak_labels import prepare_weak_pairs, remove_human_conflicts, sample_weak_training


class CategoryBatchSampler:
    def __init__(self, frame: pd.DataFrame, batch_size: int, seed: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.frame = frame.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        batches = []
        for _, group in self.frame.groupby("_cat_id", sort=True):
            idx = group.index.to_numpy(copy=True)
            rng.shuffle(idx)
            batches.extend(
                idx[start : start + self.batch_size].tolist()
                for start in range(0, len(idx), self.batch_size)
            )
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return int(sum(np.ceil(len(g) / self.batch_size) for _, g in self.frame.groupby("_cat_id")))


def train_fold(
    *, human_items_path: Path, full_items_path: Path, matches_path: Path,
    weak_matches_path: Path, manifest_path: Path, base_oof_path: Path,
    model_dir: Path, output_dir: Path, expected_split_sha: str, held_fold: int,
    device: str = "mps", weak_presample_rows: int = 180_000,
    weak_final_rows: int = 100_000, physical_batch_size: int = 8,
    effective_batch_size: int = 32, max_length: int = 128,
    last_n_layers: int = 4, max_steps: int = 800, learning_rate: float = 2e-5,
    ranking_weight: float = 0.25, seed: int = 2026,
) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_sha256(manifest) != expected_split_sha:
        raise ValueError("sealed split SHA mismatch")
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, folds = development_rows_and_folds(manifest, total_rows=len(matches))
    human_items = pd.read_parquet(human_items_path, columns=["id", "name", "attributes", "category"])
    dev = attach_pair_category(matches.iloc[dev_rows].reset_index(drop=True), human_items)
    base = pd.read_parquet(base_oof_path, columns=["row_index", "score"]).sort_values("row_index")
    if base.row_index.astype(np.int64).tolist() != dev_rows.tolist():
        raise ValueError("base OOF mismatch")
    base_scores = base.score.to_numpy(float)

    human_train = select_fold_contrastive_pairs(
        dev, folds, base_scores, held_fold=held_fold,
        max_negative_to_positive=2.0, hard_negative_fraction=0.5, seed=seed,
    )[["id1", "id2", "target", "category"]].copy()
    human_train["source"] = "human"
    human_train["weak_weight"] = 1.0
    held_mask = folds == held_fold
    held = dev.loc[held_mask].reset_index(drop=True)
    held_rows = dev_rows[held_mask]

    forbidden = forbidden_weak_item_ids(matches, manifest, held_fold=held_fold)
    weak, weak_input = prefilter_weak_candidates_parquet(
        weak_matches_path, validation_item_ids=forbidden,
        max_presample_rows=weak_presample_rows, seed=seed + held_fold,
    )
    weak, prep = prepare_weak_pairs(weak[["id1", "id2", "target"]])
    weak, conflicts = remove_human_conflicts(weak, human_train[["id1", "id2", "target"]])
    weak_ids = set(weak.id1) | set(weak.id2)
    weak_items = select_items_by_ids(full_items_path, weak_ids, include_attributes=True)
    weak = attach_pair_category(weak, weak_items)
    weak = sample_weak_training(weak, max_rows=weak_final_rows, seed=seed + held_fold)
    final_weak_ids = set(weak.id1) | set(weak.id2)
    if final_weak_ids & forbidden:
        raise RuntimeError("held/gold item leaked into weak teacher")
    weak = weak[["id1", "id2", "target", "category", "weak_weight"]].copy()
    weak["source"] = "weak"
    pairs = pd.concat([human_train, weak], ignore_index=True)

    needed = set(pairs.id1) | set(pairs.id2) | set(held.id1) | set(held.id2)
    human_subset = human_items[human_items.id.isin(needed)].copy()
    missing = needed - set(human_subset.id)
    extra = select_items_by_ids(full_items_path, missing, include_attributes=True) if missing else human_subset.iloc[:0].copy()
    items = pd.concat([human_subset, extra], ignore_index=True).drop_duplicates("id", keep="first")
    cache = normalize_items(items)
    texts = {
        item_id: f"[CAT] {norm.category}\n{serialize_item_v5(norm, max_chars=850)}"
        for item_id, norm in cache.items()
    }
    if not needed <= set(texts):
        raise RuntimeError("missing teacher text")

    categories = sorted(pairs.category.astype(str).unique())
    cat_to_id = {cat: i for i, cat in enumerate(categories)}
    pairs["_cat_id"] = pairs.category.astype(str).map(cat_to_id).astype(int)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True, num_labels=1, ignore_mismatched_sizes=True
    ).to(device)
    for parameter in model.parameters(): parameter.requires_grad = False
    layers = _find_bert_layers(model)
    for layer_idx in teacher_trainable_layer_indices(len(layers), last_n=last_n_layers):
        for parameter in layers[layer_idx].parameters(): parameter.requires_grad = True
    for name, parameter in model.named_parameters():
        if name.startswith("classifier") or ".classifier." in name or "pooler" in name or name.endswith("LayerNorm.weight") or name.endswith("LayerNorm.bias"):
            parameter.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    accumulation_steps = int(np.ceil(effective_batch_size / physical_batch_size))

    class PairDataset(Dataset):
        def __init__(self, frame): self.frame = frame.reset_index(drop=True)
        def __len__(self): return len(self.frame)
        def __getitem__(self, idx):
            row = self.frame.iloc[idx]
            return row["id1"], row["id2"], float(row["target"]), str(row["source"]), float(row["weak_weight"]), int(row["_cat_id"])

    def collate(batch):
        left, right, target, source, confidence, cat = zip(*batch)
        tokens = tokenizer(
            [texts[x] for x in left], [texts[x] for x in right],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        weights = source_loss_weights(np.asarray(source), np.asarray(confidence), human_weight=1.0, weak_scale=0.1)
        return tokens, torch.tensor(target, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32), torch.tensor(cat, dtype=torch.long)

    dataset = PairDataset(pairs)
    loader = DataLoader(
        dataset, batch_sampler=CategoryBatchSampler(pairs, physical_batch_size, seed + held_fold),
        num_workers=0, collate_fn=collate,
    )
    model.train(); optimizer.zero_grad(set_to_none=True); steps = 0; losses = []
    while steps < max_steps:
        accumulated = 0
        for batch_idx, (tokens, target, weights, cat_ids) in enumerate(loader):
            tokens = {k: v.to(device) for k, v in tokens.items()}
            target, weights, cat_ids = target.to(device), weights.to(device), cat_ids.to(device)
            logits = model(**tokens).logits.squeeze(-1)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            bce = (bce * weights).sum() / weights.sum().clamp_min(1e-6)
            rank = torch_category_ranking_loss(logits, target, cat_ids)
            loss = bce + ranking_weight * rank
            (loss / accumulation_steps).backward(); losses.append(float(loss.detach().cpu())); accumulated += 1
            if teacher_accumulation_should_step(accumulated, accumulation_steps=accumulation_steps, is_last_microbatch=batch_idx == len(loader)-1):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True); steps += 1; accumulated = 0
                if steps >= max_steps: break

    model.eval(); predictions = []
    with torch.no_grad():
        for start in range(0, len(held), 48):
            chunk = held.iloc[start:start+48]
            tokens = tokenizer([texts[x] for x in chunk.id1], [texts[x] for x in chunk.id2], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            predictions.append(torch.sigmoid(model(**tokens).logits.squeeze(-1)).cpu().numpy())
    score = np.concatenate(predictions)
    report = macro_ap_report(held, score)
    pd.DataFrame({"row_index": held_rows, "fold": held_fold, "teacher2_score": score}).sort_values("row_index").to_parquet(output_dir / f"v5g-teacher2-fold-{held_fold}-oof.parquet", index=False)
    payload = {
        "version":"v5g-field-aware-weak-ranking-teacher","held_fold":held_fold,
        "gold_metric_opened":False,"gold_rows_used":0,"human_rows":len(human_train),"weak_rows":len(weak),
        "weak_input_rows":int(weak_input),"weak_prepare":prep,"weak_conflicts":conflicts,"steps":steps,
        "macro_average_precision":report["macro_average_precision"],"per_category_ap":report["per_category_ap"],
        "elapsed_seconds":time.perf_counter()-started,
    }
    (output_dir / f"v5g-teacher2-fold-{held_fold}-metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main():
    p=argparse.ArgumentParser()
    for name in ("human-items","full-items","matches","weak-matches","manifest","base-oof","model-dir","output-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--expected-split-sha",required=True);p.add_argument("--held-fold",type=int,required=True)
    a=p.parse_args();print(json.dumps(train_fold(human_items_path=a.human_items,full_items_path=a.full_items,matches_path=a.matches,weak_matches_path=a.weak_matches,manifest_path=a.manifest,base_oof_path=a.base_oof,model_dir=a.model_dir,output_dir=a.output_dir,expected_split_sha=a.expected_split_sha,held_fold=a.held_fold),ensure_ascii=False,sort_keys=True))

if __name__=="__main__": main()
