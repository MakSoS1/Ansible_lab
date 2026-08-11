import pandas as pd

from ecup_matching.ml.v5_validation import (
    build_v5_split_manifest,
    manifest_sha256,
    validate_manifest_no_overlap,
)


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id1": [1, 2, 10, 20, 21, 30, 40, 41, 50, 60, 61, 70],
            "id2": [2, 3, 11, 21, 22, 31, 41, 42, 51, 61, 62, 71],
            "target": [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
            "category": ["a", "a", "a", "b", "b", "b", "a", "a", "b", "b", "b", "a"],
        }
    )


def test_v5_manifest_is_deterministic_complete_and_item_disjoint():
    pairs = _pairs()
    descriptors = pairs[["category", "target"]].copy()
    descriptors["difficulty_bin"] = ["easy", "hard"] * 6

    manifest = build_v5_split_manifest(
        pairs,
        descriptors,
        gold_fraction=0.25,
        n_folds=3,
        seed=2026,
    )
    manifest2 = build_v5_split_manifest(
        pairs,
        descriptors,
        gold_fraction=0.25,
        n_folds=3,
        seed=2026,
    )

    assert manifest == manifest2
    assert manifest_sha256(manifest) == manifest_sha256(manifest2)

    report = validate_manifest_no_overlap(pairs, manifest)
    assert report["row_coverage"] == len(pairs)
    assert report["duplicate_rows"] == 0
    assert report["cross_split_item_overlap"] == 0
    assert len(manifest["gold_rows"]) > 0
    assert len(manifest["fold_rows"]) == 3
    assert all(len(rows) > 0 for rows in manifest["fold_rows"])


def test_v5_manifest_keeps_connected_components_indivisible():
    pairs = _pairs()
    descriptors = pairs[["category", "target"]].copy()
    descriptors["difficulty_bin"] = "all"

    manifest = build_v5_split_manifest(
        pairs,
        descriptors,
        gold_fraction=0.25,
        n_folds=3,
        seed=2026,
    )

    split_by_row = {}
    for row in manifest["gold_rows"]:
        split_by_row[row] = "gold"
    for fold_id, rows in enumerate(manifest["fold_rows"]):
        for row in rows:
            split_by_row[row] = f"fold-{fold_id}"

    assert split_by_row[0] == split_by_row[1]  # 1-2-3 component
    assert split_by_row[3] == split_by_row[4]  # 20-21-22 component
    assert split_by_row[6] == split_by_row[7]  # 40-41-42 component
    assert split_by_row[9] == split_by_row[10]  # 60-61-62 component


def test_v5_balancer_spreads_category_and_target_strata_across_every_split():
    rows = []
    item_id = 1000
    for category in ("a", "b"):
        for target in (0, 1):
            for _ in range(4):
                rows.append(
                    {
                        "id1": item_id,
                        "id2": item_id + 1,
                        "target": target,
                        "category": category,
                    }
                )
                item_id += 10
    pairs = pd.DataFrame(rows)
    descriptors = pairs[["category", "target"]].copy()
    descriptors["difficulty_bin"] = ["easy", "hard"] * 8

    manifest = build_v5_split_manifest(
        pairs,
        descriptors,
        gold_fraction=0.25,
        n_folds=3,
        seed=2026,
    )

    for split_rows in [manifest["gold_rows"], *manifest["fold_rows"]]:
        subset = pairs.iloc[split_rows]
        assert set(subset["category"]) == {"a", "b"}
        assert set(subset["target"]) == {0, 1}
