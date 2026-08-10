from pathlib import Path

import pandas as pd

from ecup_matching.v3_prepare import prepare_v3_human_only_data


def test_prepare_v3_human_only_data_avoids_full_item_and_weak_inputs(tmp_path: Path):
    items = pd.DataFrame(
        {
            "id": list(range(1, 25)),
            "name": [f"product {i}" for i in range(1, 25)],
            "attributes": ["{}"] * 24,
            "category": ["Электроника"] * 12 + ["Аптека"] * 12,
        }
    )
    pairs = pd.DataFrame(
        {
            "id1": list(range(1, 13)),
            "id2": list(range(13, 25)),
            "target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    items_path = tmp_path / "items_human.parquet"
    matches_path = tmp_path / "matches.parquet"
    output = tmp_path / "prepared"
    items.to_parquet(items_path, index=False)
    pairs.to_parquet(matches_path, index=False)

    manifest = prepare_v3_human_only_data(
        human_items_path=items_path,
        human_matches_path=matches_path,
        output_dir=output,
        max_train_rows=8,
        max_attrs=4,
        max_chars=128,
    )

    train = pd.read_parquet(output / "train_examples.parquet")
    valid = pd.read_parquet(output / "validation_examples.parquet")
    assert manifest["mode"] == "human-only"
    assert manifest["weak_rows"] == 0
    assert manifest["validation_item_overlap"] == 0
    assert len(train) <= 8
    assert len(valid) > 0
    assert (train["source"] == "human").all()
    assert (valid["source"] == "human").all()
    assert train[["text_a", "text_b"]].notna().all().all()
