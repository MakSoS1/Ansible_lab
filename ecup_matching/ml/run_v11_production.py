from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ecup_matching.ml.data_subset import select_items_by_ids
from ecup_matching.ml.v7_frozen_split import IMMUTABLE_SPLIT_SHA, load_frozen_split_manifest, validate_frozen_split_against_matches
from ecup_matching.ml.v11_fastlex import build_fast_pair_features
from ecup_matching.ml.v11_sparse import SparseConfig, sparse_pair_scores
from ecup_matching.ml.v11_stack import fit_hgb_bundle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matches", type=Path, required=True)
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--metadata-output", type=Path, required=True)
    p.add_argument("--strict-oof", type=float, required=True)
    p.add_argument("--graph-oof", type=float, required=True)
    a = p.parse_args()
    started = time.perf_counter()

    matches = pd.read_parquet(a.matches, columns=["id1", "id2", "target"])
    manifest = load_frozen_split_manifest()
    validate_frozen_split_against_matches(matches, manifest)
    dev_rows = [row for rows in manifest["fold_rows"] for row in rows]
    dev = matches.iloc[dev_rows].reset_index(drop=True)
    needed = pd.unique(pd.concat([dev.id1, dev.id2], ignore_index=True))
    items = select_items_by_ids(a.items, needed, include_attributes=True)
    features = build_fast_pair_features(items, dev[["id1", "id2"]])
    features["sparse_cosine"] = sparse_pair_scores(
        items, dev[["id1", "id2"]], config=SparseConfig(n_features=65536)
    )
    bundle = fit_hgb_bundle(features, dev.target.to_numpy(np.int8), min_local_rows=1200, local_blend=0.35)
    a.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, a.model_output, compress=3)
    metadata = {
        "version": "v11-fastlex-1",
        "strict_oof_macro_ap": float(a.strict_oof),
        "strict_graph_oof_macro_ap": float(a.graph_oof),
        "split_sha256": IMMUTABLE_SPLIT_SHA,
        "development_rows": len(dev),
        "production_refit_uses_all_development_labels": True,
        "production_refit_score_is_not_validation": True,
        "sealed_gold_evaluated": False,
        "runtime_neural_models": False,
        "runtime_cuda_required": False,
        "sparse_n_features": 65536,
        "elapsed_seconds": time.perf_counter() - started,
    }
    a.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    a.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("V11_PRODUCTION=" + json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
