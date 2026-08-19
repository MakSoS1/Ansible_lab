import numpy as np
import pandas as pd

from ecup_matching.ml.v5_attribute_stack import crossfit_attribute_evidence_stack


def _synthetic_items_and_pairs():
    items = []
    pairs = []
    row = 0
    item = 1000
    # Every fold/category gets independent items. Name is intentionally useless;
    # identity is carried by the model/memory attributes.
    for fold in range(5):
        for category in ("electronics", "shoes"):
            for rep in range(12):
                left = item
                same = item + 1
                conflict = item + 2
                model = f"M{fold}-{rep}"
                items.extend(
                    [
                        {"id": left, "name": "товар", "category": category, "attributes": f'{{"model":"{model}","size":"42","memory":"256"}}'},
                        {"id": same, "name": "товар", "category": category, "attributes": f'{{"model":"{model}","size":"42","memory":"256"}}'},
                        {"id": conflict, "name": "товар", "category": category, "attributes": f'{{"model":"{model}-x","size":"43","memory":"128"}}'},
                    ]
                )
                pairs.append({"id1": left, "id2": same, "target": 1, "category": category, "fold": fold})
                pairs.append({"id1": left, "id2": conflict, "target": 0, "category": category, "fold": fold})
                item += 10
                row += 2
    return pd.DataFrame(items), pd.DataFrame(pairs)


def test_crossfit_attribute_stack_is_oof_and_improves_uninformative_anchor():
    items, frame = _synthetic_items_and_pairs()
    base = np.full(len(frame), 0.5, dtype=float)

    result = crossfit_attribute_evidence_stack(
        items,
        frame,
        base,
        frame["fold"].to_numpy(),
        min_support=2,
        seed=2026,
    )

    assert result["scores"].shape == (len(frame),)
    assert np.isfinite(result["scores"]).all()
    assert result["macro_average_precision"] > 0.95
    assert result["macro_average_precision"] > result["base_macro_average_precision"]
    assert len(result["fold_reports"]) == 5
    assert all(r["train_rows"] == len(frame) - len(frame) // 5 for r in result["fold_reports"])


def test_crossfit_attribute_stack_never_accepts_one_fold():
    items, frame = _synthetic_items_and_pairs()
    try:
        crossfit_attribute_evidence_stack(
            items,
            frame.iloc[:20].reset_index(drop=True),
            np.full(20, 0.5),
            np.zeros(20, dtype=int),
            min_support=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("one-fold attribute stack must be rejected")
