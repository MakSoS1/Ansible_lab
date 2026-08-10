import numpy as np

from ecup_matching.ml.blend_v2 import apply_blend, select_global_blend


def test_apply_blend_endpoints_and_range():
    structured = np.array([0.1, 0.9, 0.4])
    neural = np.array([0.8, 0.2, 0.6])
    assert np.allclose(apply_blend(structured, neural, alpha=0.0), structured)
    assert np.allclose(apply_blend(structured, neural, alpha=1.0), neural)
    mixed = apply_blend(structured, neural, alpha=0.25)
    assert np.allclose(mixed, 0.75 * structured + 0.25 * neural)
    assert ((mixed >= 0) & (mixed <= 1)).all()


def test_select_global_blend_can_choose_endpoint_or_mixture():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    categories = np.array(["a"] * 4 + ["b"] * 4)
    structured = np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.4, 0.7, 0.3])
    neural = np.array([0.4, 0.6, 0.1, 0.9, 0.2, 0.8, 0.1, 0.9])
    result = select_global_blend(
        structured,
        neural,
        y,
        categories,
        alphas=np.linspace(0.0, 1.0, 11),
    )
    assert 0.0 <= result["alpha"] <= 1.0
    assert 0.0 <= result["macro_average_precision"] <= 1.0
    assert len(result["grid"]) == 11
    assert len(result["per_category_ap"]) == 2


def test_select_global_blend_rejects_misaligned_inputs():
    try:
        select_global_blend(
            np.array([0.1, 0.2]),
            np.array([0.3]),
            np.array([0, 1]),
            np.array(["a", "a"]),
        )
        assert False, "expected length mismatch"
    except ValueError:
        pass
