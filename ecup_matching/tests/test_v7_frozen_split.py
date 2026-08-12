import json
from pathlib import Path

import pandas as pd
import pytest

from ecup_matching.ml.v7_frozen_split import (
    DEFAULT_FROZEN_SPLIT_PATH,
    IMMUTABLE_SPLIT_SHA,
    load_frozen_split_manifest,
    validate_frozen_split_against_matches,
)


def test_committed_frozen_manifest_is_exact_v5_v6_validation_artifact():
    manifest = load_frozen_split_manifest()

    assert DEFAULT_FROZEN_SPLIT_PATH.is_file()
    assert manifest["version"] == 1
    assert manifest["seed"] == 2026
    assert manifest["row_count"] == 365_654
    assert manifest["component_count"] == 345_654
    assert manifest["n_folds"] == 5
    assert len(manifest["gold_rows"]) == 80_444
    assert sum(map(len, manifest["fold_rows"])) == 285_210


def test_frozen_manifest_rejects_any_content_drift(tmp_path):
    manifest = load_frozen_split_manifest()
    manifest["fold_rows"][0][0], manifest["fold_rows"][0][1] = (
        manifest["fold_rows"][0][1],
        manifest["fold_rows"][0][0],
    )
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable split SHA mismatch"):
        load_frozen_split_manifest(path)


def test_frozen_manifest_validation_checks_item_overlap_not_evolving_features(tmp_path):
    manifest = {
        "version": 1,
        "seed": 2026,
        "gold_fraction": 0.22,
        "n_folds": 2,
        "row_count": 3,
        "component_count": 3,
        "descriptor_names": ["irrelevant_to_loading"],
        "gold_rows": [0],
        "fold_rows": [[1], [2]],
    }
    matches = pd.DataFrame(
        {
            "id1": [1, 3, 5],
            "id2": [2, 4, 6],
            "target": [1, 0, 1],
        }
    )
    report = validate_frozen_split_against_matches(matches, manifest)
    assert report == {
        "row_coverage": 3,
        "duplicate_rows": 0,
        "missing_rows": 0,
        "cross_split_item_overlap": 0,
    }

    leaked = matches.copy()
    leaked.loc[2, "id1"] = 1
    with pytest.raises(ValueError, match="cross-split item overlap"):
        validate_frozen_split_against_matches(leaked, manifest)


def test_expected_sha_constant_never_tracks_recomputed_feature_split():
    assert IMMUTABLE_SPLIT_SHA == "aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b"
