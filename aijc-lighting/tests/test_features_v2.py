import numpy as np
from PIL import Image

from src.features_v2 import extract_v2_feature_frame, extract_v2_features


def _save_constant(path, value):
    Image.fromarray(np.full((64, 64, 3), value, dtype=np.uint8)).save(path)


def test_v2_features_are_finite_and_core_luminance_monotonic(tmp_path):
    black = tmp_path / "black.png"
    gray = tmp_path / "gray.png"
    white = tmp_path / "white.png"
    _save_constant(black, 5)
    _save_constant(gray, 128)
    _save_constant(white, 250)

    rows = [extract_v2_features(path) for path in (black, gray, white)]
    keys = list(rows[0])
    assert keys == list(rows[1]) == list(rows[2])
    values = np.asarray([[row[key] for key in keys] for row in rows], dtype=float)
    assert np.isfinite(values).all()
    assert rows[0]["v2_lum_mean"] < rows[1]["v2_lum_mean"] < rows[2]["v2_lum_mean"]
    assert rows[0]["v2_shadow_frac_32"] > rows[1]["v2_shadow_frac_32"]
    assert rows[2]["v2_highlight_frac_224"] > rows[1]["v2_highlight_frac_224"]


def test_v2_frame_has_stable_sorted_columns(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _save_constant(a, 30)
    _save_constant(b, 220)
    frame = extract_v2_feature_frame([a, b])
    assert frame.shape[0] == 2
    assert list(frame.columns) == sorted(frame.columns)
    assert np.isfinite(frame.to_numpy()).all()
