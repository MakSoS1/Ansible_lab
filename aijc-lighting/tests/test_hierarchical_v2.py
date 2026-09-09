import numpy as np

from src.hierarchical import compose_three_class_probs


def test_compose_three_class_probs_is_normalized():
    p_dark = np.array([0.8, 0.1])
    p_bright = np.array([0.25, 0.75])
    out = compose_three_class_probs(p_dark, p_bright)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out.sum(1), 1.0)
    np.testing.assert_allclose(out[0], [0.8, 0.15, 0.05])


def test_compose_clips_invalid_probabilities():
    out = compose_three_class_probs(np.array([-1.0, 2.0]), np.array([2.0, -1.0]))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(1), 1.0)
    assert (out >= 0).all()
