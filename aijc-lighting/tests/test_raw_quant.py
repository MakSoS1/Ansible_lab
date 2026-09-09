import numpy as np
from PIL import Image

from src.raw_quant import extract_raw_quant_features


def test_raw_quant_detects_level_compression(tmp_path):
    x = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    rgb = np.stack([x, x, x], axis=-1)
    normal = tmp_path / 'normal.png'
    dark = tmp_path / 'dark.png'
    Image.fromarray(rgb).save(normal)
    Image.fromarray((rgb.astype(np.float32) * 0.5).round().astype(np.uint8)).save(dark)
    a = extract_raw_quant_features(normal)
    b = extract_raw_quant_features(dark)
    assert a['rq_r_unique'] > b['rq_r_unique']
    assert b['rq_r_zero_bins'] > a['rq_r_zero_bins']
    assert b['rq_r_scale_fit_0_5'] > b['rq_r_scale_fit_1_0']


def test_raw_quant_is_finite(tmp_path):
    rng = np.random.default_rng(42)
    p = tmp_path / 'random.png'
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(p)
    f = extract_raw_quant_features(p)
    assert len(f) > 100
    assert np.isfinite(np.fromiter(f.values(), dtype=float)).all()
