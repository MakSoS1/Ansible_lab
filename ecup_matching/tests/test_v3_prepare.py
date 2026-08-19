import pandas as pd

from ecup_matching.v3_prepare import compact_serialized_examples, prepared_remote_prefix


def test_compact_serialized_examples_preserves_text_source_and_human_positives():
    rows = [
        {"id1": 1, "id2": 2, "target": 1.0, "category": "Электроника", "sample_weight": 3.0, "text_a": "a1", "text_b": "b1", "source": "human"},
        {"id1": 3, "id2": 4, "target": 1.0, "category": "Аптека", "sample_weight": 3.0, "text_a": "a2", "text_b": "b2", "source": "human"},
    ]
    for i in range(12):
        rows.append({
            "id1": 100 + i * 2,
            "id2": 101 + i * 2,
            "target": 0.0 if i % 2 == 0 else 0.99,
            "category": "Электроника" if i < 8 else "Аптека",
            "sample_weight": 1.0,
            "text_a": f"a{i+10}",
            "text_b": f"b{i+10}",
            "source": "human" if i < 6 else "weak",
        })
    frame = pd.DataFrame(rows)
    expected_human_positive_pairs = set(
        map(
            tuple,
            frame.loc[
                (frame["source"] == "human") & (frame["target"].astype(float) >= 0.5),
                ["id1", "id2"],
            ].to_numpy(),
        )
    )

    compact = compact_serialized_examples(
        frame,
        max_rows=10,
        priority_categories={"Электроника"},
        priority_fraction=0.60,
        seed=2026,
    )

    assert len(compact) == 10
    assert {"text_a", "text_b", "source", "sample_weight"}.issubset(compact.columns)
    human_pos = compact[(compact["source"] == "human") & (compact["target"] >= 0.5)]
    assert set(map(tuple, human_pos[["id1", "id2"]].to_numpy())) == expected_human_positive_pairs
    assert compact[["text_a", "text_b"]].notna().all().all()


def test_prepared_remote_prefix_is_private_path_and_commit_scoped():
    assert prepared_remote_prefix("abcdef123456789") == "experiments/v3/prepared/abcdef123456"
