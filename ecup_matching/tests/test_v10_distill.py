import numpy as np
import pytest

from ecup_matching.ml.v10_distill import blend_hard_teacher_targets


def test_blend_hard_teacher_targets_is_convex_and_keeps_hard_endpoint_at_zero_weight():
    hard = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    teacher = np.asarray([0.2, 0.8, 0.4, 0.9], dtype=np.float64)

    np.testing.assert_allclose(
        blend_hard_teacher_targets(hard, teacher, teacher_weight=0.25),
        np.asarray([0.05, 0.95, 0.85, 0.225]),
    )
    np.testing.assert_array_equal(
        blend_hard_teacher_targets(hard, teacher, teacher_weight=0.0),
        hard,
    )


def test_blend_hard_teacher_targets_supports_hard_label_only_validation():
    hard = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    np.testing.assert_array_equal(
        blend_hard_teacher_targets(hard, None, teacher_weight=0.0),
        hard,
    )


def test_blend_hard_teacher_targets_rejects_invalid_teacher_or_weight():
    hard = np.asarray([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="teacher_weight"):
        blend_hard_teacher_targets(hard, np.asarray([0.2, 0.8]), teacher_weight=1.1)
    with pytest.raises(ValueError, match="length"):
        blend_hard_teacher_targets(hard, np.asarray([0.2]), teacher_weight=0.2)
    with pytest.raises(ValueError, match="\[0, 1\]"):
        blend_hard_teacher_targets(hard, np.asarray([0.2, 1.2]), teacher_weight=0.2)
    with pytest.raises(ValueError, match="binary"):
        blend_hard_teacher_targets(np.asarray([0.1, 1.0]), np.asarray([0.2, 0.8]), teacher_weight=0.2)
