from pathlib import Path

import numpy as np
from PIL import Image

from src.features import extract_global_features, extract_spatial_features, extract_feature_table


def _save(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), 'RGB').save(path)


def test_global_luminance_features_are_monotonic(tmp_path: Path):
    paths = []
    for name, value in [('black', 0), ('gray', 128), ('white', 255)]:
        p = tmp_path / f'{name}.png'
        _save(p, np.full((32, 32, 3), value, dtype=np.uint8))
        paths.append(p)
    feats = [extract_global_features(p) for p in paths]
    assert feats[0]['lum_mean'] < feats[1]['lum_mean'] < feats[2]['lum_mean']
    assert feats[0]['black_frac_5'] > 0.99
    assert feats[2]['white_frac_250'] > 0.99
    assert abs(sum(feats[1][f'lum_hist_{i:02d}'] for i in range(32)) - 1.0) < 1e-6
    assert all(np.isfinite(list(f.values())).all() for f in feats)


def test_spatial_center_edge_contrast(tmp_path: Path):
    arr = np.full((64, 64, 3), 20, dtype=np.uint8)
    arr[16:48, 16:48] = 230
    p = tmp_path / 'center.png'
    _save(p, arr)
    f = extract_spatial_features(p)
    assert f['center_minus_edge'] > 0.3
    assert 'zone4_mean_00' in f
    assert 'zone8_mean_63' in f
    assert f['wavelet_h_energy'] >= 0
    assert f['wavelet_v_energy'] >= 0
    assert f['wavelet_d_energy'] >= 0


def test_feature_table_has_deterministic_columns(tmp_path: Path):
    p1 = tmp_path / 'a.png'
    p2 = tmp_path / 'b.png'
    _save(p1, np.full((16, 16, 3), 30, dtype=np.uint8))
    _save(p2, np.full((16, 16, 3), 220, dtype=np.uint8))
    table1 = extract_feature_table([p1, p2], mode='all')
    table2 = extract_feature_table([p1, p2], mode='all')
    assert table1.columns.tolist() == table2.columns.tolist()
    assert np.allclose(table1.values, table2.values)
