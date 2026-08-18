from __future__ import annotations

from ecup_matching.ml.v18_views import (
    augment_serialized_view,
    deterministic_decision,
)


TEXT = "\n".join(
    [
        "[CAT] phones",
        "[NAME] Example Phone 128GB",
        "[BRAND] Example",
        "[MODEL] ex128",
        "[IDENTITY] storage=storage_bytes_137438953472",
        "[NUMERIC] storage_137438953472 | 128",
        "[RESIDUAL] warranty=2 years | package=box",
    ]
)


def test_view_drops_only_requested_low_priority_lines() -> None:
    out = augment_serialized_view(TEXT, drop_residual=True, drop_numeric=True)
    assert "[RESIDUAL]" not in out
    assert "[NUMERIC]" not in out
    for protected in ("[CAT]", "[NAME]", "[BRAND]", "[MODEL]", "[IDENTITY]"):
        assert protected in out


def test_numeric_is_kept_without_model_or_identity_evidence() -> None:
    text = "[CAT] misc\n[NAME] Plain thing\n[NUMERIC] count_2\n[RESIDUAL] foo=bar"
    out = augment_serialized_view(text, drop_residual=False, drop_numeric=True)
    assert "[NUMERIC] count_2" in out


def test_deterministic_decision_is_reproducible_and_epoch_sensitive() -> None:
    first = [deterministic_decision(seed=2026, epoch=0, index=i, stream="swap", probability=0.5) for i in range(64)]
    again = [deterministic_decision(seed=2026, epoch=0, index=i, stream="swap", probability=0.5) for i in range(64)]
    next_epoch = [deterministic_decision(seed=2026, epoch=1, index=i, stream="swap", probability=0.5) for i in range(64)]
    assert first == again
    assert first != next_epoch
    assert any(first) and not all(first)


def test_decision_probability_boundaries() -> None:
    assert not deterministic_decision(seed=1, epoch=1, index=1, stream="x", probability=0.0)
    assert deterministic_decision(seed=1, epoch=1, index=1, stream="x", probability=1.0)
