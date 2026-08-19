from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


def test_no_teacher_composition_never_requires_teacher() -> None:
    from ecup_matching.submission.predict_v10_faststack import compose_no_teacher_signals

    scores = {
        "weak": np.array([0.1, 0.2, 0.3]),
        "sparse": np.array([0.3, 0.2, 0.1]),
        "explicit": np.array([0.2, 0.4, 0.6]),
        "contrastive": np.array([0.7, 0.5, 0.3]),
        "typed_explicit": np.array([0.4, 0.6, 0.8]),
    }
    six = compose_no_teacher_signals(scores)
    assert set(six) == {"weak", "sparse", "explicit", "contrastive", "teacher", "typed_explicit"}
    assert np.isfinite(six["teacher"]).all()
    assert np.allclose(
        six["teacher"],
        np.mean(
            np.column_stack([
                np.array([0.0, 0.5, 1.0]),
                np.array([1.0, 0.5, 0.0]),
                np.array([0.0, 0.5, 1.0]),
                np.array([1.0, 0.5, 0.0]),
                np.array([0.0, 0.5, 1.0]),
            ]),
            axis=1,
        ),
    )


def test_faststack_runtime_has_no_teacher_checkpoint_parameter() -> None:
    import inspect
    from ecup_matching.submission.predict_v10_faststack import predict_to_csv_v10_faststack

    params = set(inspect.signature(predict_to_csv_v10_faststack).parameters)
    assert "teacher_model_dir" not in params
    assert "teacher_model_path" not in params


def test_faststack_package_guard_rejects_teacher_assets(tmp_path: Path) -> None:
    from ecup_matching.submission.predict_v10_faststack import assert_no_teacher_assets

    assert_no_teacher_assets(tmp_path)
    (tmp_path / "model_v5_teacher").mkdir()
    try:
        assert_no_teacher_assets(tmp_path)
    except RuntimeError as exc:
        assert "teacher" in str(exc).lower()
    else:
        raise AssertionError("teacher checkpoint must be rejected")


def test_contrastive_only_cache_matches_legacy_contrastive_view() -> None:
    from ecup_matching.submission.v10_text_cache import build_contrastive_text_cache

    class FakeNorm:
        @staticmethod
        def normalize_item(item_id, name, attributes, category):
            return SimpleNamespace(
                item_id=item_id,
                name=str(name),
                attributes=str(attributes),
                category=str(category),
            )

    class FakeText:
        @staticmethod
        def serialize_item_v5(norm, *, max_chars):
            return f"{norm.item_id}|{norm.category}|{norm.name}|{norm.attributes}"[:max_chars]

    items = pd.DataFrame(
        {
            "id": [11, 12, 13],
            "name": ["alpha", "beta", "gamma"],
            "attributes": ["a=1", "b=2", "c=3"],
            "category": ["x", "y", "z"],
        }
    )
    got = build_contrastive_text_cache(items, FakeNorm, FakeText, workers=1)
    expected = {
        row.id: FakeText.serialize_item_v5(
            FakeNorm.normalize_item(row.id, row.name, row.attributes, row.category),
            max_chars=700,
        )
        for row in items.itertuples(index=False)
    }
    assert got == expected
