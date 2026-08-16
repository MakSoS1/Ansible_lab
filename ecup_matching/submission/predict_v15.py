from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd

from ecup_matching.v15_fields import normalize_item_fields
from ecup_matching.v15_model import V15Matcher
from ecup_matching.v15_pair_features import PAIR_FEATURE_NAMES, build_pair_features


def _item_text(item, include_attributes: bool) -> str:
    parts = [f"[TITLE] {item.title}"]
    if item.category:
        parts.append(f"[CATEGORY] {item.category}")
    if include_attributes:
        if item.brand:
            parts.append(f"[BRAND] {item.brand}")
        if item.model_tokens:
            parts.append("[MODEL] " + " | ".join(item.model_tokens))
        if item.numeric_tokens:
            parts.append("[NUMERIC] " + " | ".join(item.numeric_tokens))
        if item.attributes:
            attrs = item.attributes[:48]
            parts.append("[ATTR] " + " ; ".join(f"{k}={v}" for k, v in attrs))
    return " ".join(parts)


def _torch_load(path: Path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def predict_to_csv_v15(
    *,
    items_path: Path,
    matches_path: Path,
    checkpoint_path: Path,
    base_model_dir: Path,
    output_path: Path,
    batch_size: int = 64,
) -> None:
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("v15 submission requires CUDA")

    ckpt = _torch_load(Path(checkpoint_path))
    required = {
        "model_state", "variant", "max_length", "categories", "pair_feature_names",
        "include_attributes", "use_typed_features", "use_category_head",
        "base_model_revision", "base_model_weights_sha256",
    }
    missing = required - set(ckpt)
    if missing:
        raise RuntimeError(f"v15 checkpoint missing keys: {sorted(missing)}")
    if list(ckpt["pair_feature_names"]) != list(PAIR_FEATURE_NAMES):
        raise RuntimeError("v15 pair feature contract mismatch")
    max_length = int(ckpt["max_length"])
    if max_length != 128:
        raise RuntimeError("v15 runtime only accepts max_length=128")

    matches = pd.read_parquet(matches_path, columns=["id1", "id2"]).reset_index(drop=True)
    wanted = set(matches["id1"].tolist()) | set(matches["id2"].tolist())
    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    items = items.loc[items["id"].isin(wanted)].reset_index(drop=True)
    if set(items["id"].tolist()) != wanted:
        raise RuntimeError("matches reference item ids absent from items")

    norm = {}
    for r in items.itertuples(index=False):
        norm[r.id] = normalize_item_fields(r.name, r.attributes, r.category)
    include_attributes = bool(ckpt["include_attributes"])
    texts = {i: _item_text(v, include_attributes) for i, v in norm.items()}

    categories = [str(x) for x in ckpt["categories"]]
    category_to_id = {c: i for i, c in enumerate(categories)}
    use_typed = bool(ckpt["use_typed_features"])
    use_category = bool(ckpt["use_category_head"])

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True)
    cfg = AutoConfig.from_pretrained(base_model_dir, local_files_only=True)
    if hasattr(cfg, "reference_compile"):
        cfg.reference_compile = False
    backbone = AutoModel.from_config(cfg)
    model = V15Matcher(
        backbone,
        typed_feature_dim=len(PAIR_FEATURE_NAMES),
        num_categories=len(categories),
        use_typed_features=use_typed,
        use_category_head=use_category,
        dropout=0.05,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.cuda().eval()

    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(matches), int(batch_size)):
            frame = matches.iloc[start:start + int(batch_size)]
            left: list[str] = []
            right: list[str] = []
            features: list[np.ndarray] = []
            category_ids: list[int] = []
            for r in frame.itertuples(index=False):
                a, b = r.id1, r.id2
                ka = (texts[a], str(a)); kb = (texts[b], str(b))
                u, v = (a, b) if ka <= kb else (b, a)
                left.append(texts[u]); right.append(texts[v])
                if use_typed:
                    features.append(build_pair_features(norm[a], norm[b]))
                if use_category:
                    cat = str(norm[a].category)
                    cat2 = str(norm[b].category)
                    if cat != cat2:
                        raise RuntimeError("candidate pair crosses official categories")
                    if cat not in category_to_id:
                        raise RuntimeError(f"unknown category at inference: {cat!r}")
                    category_ids.append(category_to_id[cat])
            tok = tokenizer(
                left, right, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            )
            tok = {k: v.cuda(non_blocking=True) for k, v in tok.items()}
            kwargs = {}
            if use_typed:
                kwargs["typed_features"] = torch.tensor(np.stack(features), dtype=torch.float32, device="cuda")
            if use_category:
                kwargs["category_ids"] = torch.tensor(category_ids, dtype=torch.long, device="cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(**tok, **kwargs)
            scores.append(torch.sigmoid(logits).float().cpu().numpy())

    pred = np.concatenate(scores).astype(np.float64) if scores else np.empty(0, dtype=np.float64)
    if len(pred) != len(matches) or not np.isfinite(pred).all():
        raise RuntimeError("v15 produced invalid prediction vector")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = matches.copy()
    out["predict"] = pred
    out.to_csv(output_path, index=False)
