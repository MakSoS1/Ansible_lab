"""Prepare D3 authoritative teacher-audit rows and a conservative all-human forbidden-item universe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .run_v5_pretrained_biencoder import development_rows_and_folds
from .run_v7_outer_oof import IMMUTABLE_SPLIT_SHA
from .run_v7_outer_oof_frozen import _load_immutable_manifest
from .textnorm import normalize_item
from .v20_admission import build_fold_safe_audit_split
from .v20_strata import classify_pair_stratum


def run(*, human_items_path: Path, matches_path: Path, output_dir: Path, fold: int, audit_fraction: float, seed: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = pd.read_parquet(human_items_path, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    pairs, manifest, overlap = _load_immutable_manifest(items, matches, expected_split_sha=IMMUTABLE_SPLIT_SHA)
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(matches))
    if len(dev_rows) != 285_210 or len(manifest.get("gold_rows", [])) != 80_444:
        raise RuntimeError("immutable row counts changed")
    dev = pairs.iloc[dev_rows].reset_index(drop=True)
    outer_train = dev.loc[fold_ids != int(fold), ["id1", "id2", "target", "category"]].reset_index(drop=True)
    held = dev.loc[fold_ids == int(fold), ["id1", "id2", "target", "category"]].reset_index(drop=True)
    needed = set(outer_train.id1) | set(outer_train.id2)
    item_map = {
        r.id: normalize_item(r.id, r.name, r.attributes, r.category)
        for r in items.loc[items["id"].isin(needed)].itertuples(index=False)
    }
    reasons = []
    for row in outer_train.itertuples(index=False):
        s = classify_pair_stratum(item_map[row.id1], item_map[row.id2])
        reasons.append(f"{s.category}|{s.reason_code}|{s.difficulty}")
    outer_train["stratum"] = reasons
    model_train, llm_audit, audit_report = build_fold_safe_audit_split(
        outer_train, audit_fraction=audit_fraction, seed=seed + int(fold) * 1009
    )
    held_items = set(held.id1) | set(held.id2)
    for name, frame in [("model_train", model_train), ("llm_audit", llm_audit)]:
        if (set(frame.id1) | set(frame.id2)) & held_items:
            raise RuntimeError(f"{name} overlaps held fold")
    model_train.to_parquet(output_dir / "model_train_human.parquet", index=False)
    llm_audit.to_parquet(output_dir / "llm_audit_human.parquet", index=False)
    all_human_ids = sorted(set(matches.id1.astype(int)) | set(matches.id2.astype(int)))
    pd.DataFrame({"id": all_human_ids}).to_parquet(output_dir / "all_human_forbidden_ids.parquet", index=False)
    report = {
        "version": "v20-human-audit-split-v1", "fold": int(fold),
        "model_train_rows": int(len(model_train)), "audit_rows": int(len(llm_audit)),
        "held_rows": int(len(held)), "all_human_forbidden_items": int(len(all_human_ids)),
        "audit_split": audit_report, "split_sha256": IMMUTABLE_SPLIT_SHA,
        "cross_split_item_overlap": int(overlap["cross_split_item_overlap"]),
        "sealed_gold_opened": False, "gold_rows_scored": 0,
    }
    (output_dir / "audit-split.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--human-items", type=Path, required=True)
    p.add_argument("--matches", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--audit-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    run(human_items_path=a.human_items, matches_path=a.matches, output_dir=a.output_dir, fold=a.fold, audit_fraction=a.audit_fraction, seed=a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
