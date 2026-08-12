from __future__ import annotations

import pandas as pd

from . import run_v7_outer_oof as base
from .run_v7_outer_oof_fast import _load_model_no_checkpoint
from .train_v1 import attach_pair_category
from .v7_frozen_split import load_frozen_split_manifest, validate_frozen_split_against_matches


def _load_immutable_manifest(
    human_items: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    expected_split_sha: str,
):
    pairs = attach_pair_category(matches.copy(), human_items).reset_index(drop=True)
    manifest = load_frozen_split_manifest(expected_sha=expected_split_sha)
    overlap = validate_frozen_split_against_matches(matches, manifest)
    base._phase(
        "frozen-split-loaded",
        split_sha=expected_split_sha,
        development_rows=sum(len(rows) for rows in manifest["fold_rows"]),
        sealed_gold_rows=len(manifest["gold_rows"]),
        cross_split_item_overlap=overlap["cross_split_item_overlap"],
    )
    return pairs, manifest, overlap


def main() -> int:
    # Only two implementation details are replaced: validation split source and
    # training-time checkpoint recomputation. Model architecture, curriculum,
    # scoring, OOF aggregation and metric code stay in the already-tested driver.
    base._build_immutable_manifest = _load_immutable_manifest
    base._load_model = _load_model_no_checkpoint
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())