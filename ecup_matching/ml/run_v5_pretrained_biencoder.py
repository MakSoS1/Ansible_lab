from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .textnorm import normalize_item
from .v5_embeddings import EMBEDDING_PAIR_FEATURE_NAMES, build_embedding_pair_features
from .v5_evaluation import macro_ap_report
from .v5_item_text import serialize_item_v5
from .v5_semantic_stack import crossfit_semantic_stack


def development_rows_and_folds(manifest: dict, *, total_rows: int) -> tuple[np.ndarray, np.ndarray]:
    gold = [int(x) for x in manifest.get("gold_rows", [])]
    folds = manifest.get("fold_rows", [])
    if len(folds) < 2:
        raise ValueError("manifest must contain at least two development folds")
    row_to_fold: dict[int, int] = {}
    for fold_id, rows in enumerate(folds):
        for raw in rows:
            row = int(raw)
            if row in row_to_fold:
                raise ValueError("development row occurs in multiple folds")
            row_to_fold[row] = fold_id
    if set(gold) & set(row_to_fold):
        raise ValueError("gold row occurs in development folds")
    all_rows = set(gold) | set(row_to_fold)
    if all_rows != set(range(total_rows)):
        raise ValueError("manifest must cover every source row exactly once")
    rows = np.asarray(sorted(row_to_fold), dtype=np.int64)
    fold_ids = np.asarray([row_to_fold[int(row)] for row in rows], dtype=np.int16)
    return rows, fold_ids


def _encode_items(
    items: pd.DataFrame,
    *,
    model_dir: Path,
    device: str,
    batch_size: int,
    chunk_items: int,
    max_seq_length: int,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only in GPU workflow
        raise RuntimeError("sentence-transformers is required for pretrained evaluation") from exc

    model = SentenceTransformer(str(model_dir), device=device)
    model.max_seq_length = int(max_seq_length)
    dimension = int(model.get_sentence_embedding_dimension())
    output = np.empty((len(items), dimension), dtype=np.float32)

    for start in range(0, len(items), chunk_items):
        stop = min(len(items), start + chunk_items)
        chunk = items.iloc[start:stop]
        texts: list[str] = []
        for item_id, name, attributes, category in chunk[
            ["id", "name", "attributes", "category"]
        ].itertuples(index=False, name=None):
            norm = normalize_item(item_id, name, attributes, category)
            texts.append(serialize_item_v5(norm, max_chars=700))
        encoded = model.encode(
            texts,
            batch_size=int(batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        )
        output[start:stop] = np.asarray(encoded, dtype=np.float32)
        print(f"encoded_items={stop}/{len(items)}", flush=True)
        del texts, encoded
    return output


def run_pretrained_biencoder(
    *,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    base_oof_path: Path,
    model_dir: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 192,
    chunk_items: int = 20_000,
    max_seq_length: int = 96,
    expected_split_sha: str | None = None,
) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(matches))

    if expected_split_sha is not None:
        from .v5_validation import manifest_sha256

        actual = manifest_sha256(manifest)
        if actual != expected_split_sha:
            raise ValueError(f"split manifest SHA mismatch: {actual}")

    dev_pairs = matches.iloc[dev_rows].reset_index(drop=True)
    wanted_ids = pd.unique(pd.concat([dev_pairs["id1"], dev_pairs["id2"]], ignore_index=True))
    # Initial transfer rung is deliberately name-centric for speed. Passing
    # include_attributes=False preserves category and name while making attrs
    # an empty JSON object. Gold items are absent because connected components
    # make dev/gold item sets disjoint.
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=False)
    item_ids = items["id"].tolist()
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    if len(item_index) != len(item_ids):
        raise ValueError("duplicate item ids in selected development items")

    gold_rows = np.asarray(manifest["gold_rows"], dtype=np.int64)
    if len(gold_rows):
        gold_pairs = matches.iloc[gold_rows]
        gold_items = set(gold_pairs["id1"].tolist()) | set(gold_pairs["id2"].tolist())
        overlap = gold_items & set(item_ids)
        if overlap:
            raise RuntimeError(f"gold items leaked into encoder input: {len(overlap)}")

    encode_started = time.perf_counter()
    embeddings = _encode_items(
        items,
        model_dir=model_dir,
        device=device,
        batch_size=batch_size,
        chunk_items=chunk_items,
        max_seq_length=max_seq_length,
    )
    encode_seconds = time.perf_counter() - encode_started

    left = np.fromiter((item_index[x] for x in dev_pairs["id1"].tolist()), dtype=np.int64, count=len(dev_pairs))
    right = np.fromiter((item_index[x] for x in dev_pairs["id2"].tolist()), dtype=np.int64, count=len(dev_pairs))
    semantic_values = build_embedding_pair_features(embeddings[left], embeddings[right])
    semantic = pd.DataFrame(semantic_values, columns=EMBEDDING_PAIR_FEATURE_NAMES)

    category_by_id = items.set_index("id")["category"].astype(str)
    dev_pairs = dev_pairs.copy()
    dev_pairs["category"] = dev_pairs["id1"].map(category_by_id)
    if dev_pairs["category"].isna().any():
        raise RuntimeError("failed to attach category to development pairs")

    base_oof = pd.read_parquet(base_oof_path, columns=["row_index", "score"]).sort_values("row_index")
    if base_oof["row_index"].astype(np.int64).tolist() != dev_rows.tolist():
        raise ValueError("base OOF row indices do not match sealed development rows")
    base_scores = base_oof["score"].to_numpy(dtype=np.float64)

    cosine_report = macro_ap_report(dev_pairs, semantic["embedding_cosine"].to_numpy())
    stack = crossfit_semantic_stack(dev_pairs, base_scores, semantic, fold_ids, seed=2026)
    payload = {
        "version": "v5c-pretrained-biencoder",
        "model": model_dir.name,
        "device": device,
        "development_rows": int(len(dev_rows)),
        "development_items_encoded": int(len(items)),
        "gold_rows_encoded": 0,
        "gold_items_encoded": 0,
        "gold_metric_opened": False,
        "max_seq_length": int(max_seq_length),
        "batch_size": int(batch_size),
        "encode_seconds": float(encode_seconds),
        "semantic_cosine_macro_ap": float(cosine_report["macro_average_precision"]),
        "base_oof_macro_ap": float(stack["base_macro_average_precision"]),
        "stack_oof_macro_ap": float(stack["macro_average_precision"]),
        "delta_vs_base": float(stack["delta_vs_base"]),
        "fold_reports": stack["fold_reports"],
        "per_category_ap": stack["per_category_ap"],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "v5c-pretrained-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "row_index": dev_rows,
            "fold": fold_ids,
            **{name: semantic[name].to_numpy() for name in EMBEDDING_PAIR_FEATURE_NAMES},
            "score": stack["scores"],
        }
    ).to_parquet(output_dir / "v5c-pretrained-oof.parquet", index=False)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-oof", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--chunk-items", type=int, default=20_000)
    parser.add_argument("--max-seq-length", type=int, default=96)
    parser.add_argument("--expected-split-sha")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_pretrained_biencoder(
        items_path=args.items,
        matches_path=args.matches,
        manifest_path=args.manifest,
        base_oof_path=args.base_oof,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        chunk_items=args.chunk_items,
        max_seq_length=args.max_seq_length,
        expected_split_sha=args.expected_split_sha,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
