from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def _load_rgb(path: Path | str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float32)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0


def _entropy_from_hist(hist: np.ndarray) -> float:
    p = hist[hist > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def _moments(prefix: str, values: np.ndarray) -> dict[str, float]:
    flat = values.astype(np.float64).ravel()
    mean = float(flat.mean())
    std = float(flat.std())
    centered = flat - mean
    denom = std + 1e-12
    skew = float(np.mean((centered / denom) ** 3)) if std > 1e-12 else 0.0
    kurt = float(np.mean((centered / denom) ** 4) - 3.0) if std > 1e-12 else 0.0
    return {
        f'{prefix}_mean': mean,
        f'{prefix}_std': std,
        f'{prefix}_skew': skew,
        f'{prefix}_kurt': kurt,
        f'{prefix}_min': float(flat.min()),
        f'{prefix}_max': float(flat.max()),
    }


def extract_global_features(path: Path | str) -> dict[str, float]:
    rgb = _load_rgb(path)
    lum = _luminance(rgb)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    f: dict[str, float] = {}
    f.update(_moments('lum', lum))
    for p in PERCENTILES:
        f[f'lum_p{p:02d}'] = float(np.percentile(lum, p))

    hist, _ = np.histogram(lum, bins=32, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    for i, value in enumerate(hist):
        f[f'lum_hist_{i:02d}'] = float(value)
    f['lum_entropy'] = _entropy_from_hist(hist)

    raw_lum = lum * 255.0
    for threshold in (1, 5, 10, 20, 32, 48, 64):
        f[f'black_frac_{threshold}'] = float((raw_lum <= threshold).mean())
    for threshold in (192, 208, 224, 240, 245, 250, 254):
        f[f'white_frac_{threshold}'] = float((raw_lum >= threshold).mean())

    channels = {
        'r': rgb[..., 0] / 255.0,
        'g': rgb[..., 1] / 255.0,
        'b': rgb[..., 2] / 255.0,
        'h': hsv[..., 0] / 179.0,
        's': hsv[..., 1] / 255.0,
        'v': hsv[..., 2] / 255.0,
        'lab_l': lab[..., 0] / 255.0,
        'lab_a': (lab[..., 1] - 128.0) / 127.0,
        'lab_b': (lab[..., 2] - 128.0) / 127.0,
    }
    for name, values in channels.items():
        f.update(_moments(name, values))

    eps = 1e-6
    f['rg_ratio'] = float((rgb[..., 0].mean() + eps) / (rgb[..., 1].mean() + eps))
    f['rb_ratio'] = float((rgb[..., 0].mean() + eps) / (rgb[..., 2].mean() + eps))
    f['gb_ratio'] = float((rgb[..., 1].mean() + eps) / (rgb[..., 2].mean() + eps))
    f['value_minus_lum'] = float(channels['v'].mean() - lum.mean())
    f['saturation_x_value'] = float((channels['s'] * channels['v']).mean())
    f['dynamic_range_90_10'] = float(np.percentile(lum, 90) - np.percentile(lum, 10))
    f['dynamic_range_99_01'] = float(np.percentile(lum, 99) - np.percentile(lum, 1))
    f['median_minus_mean'] = float(np.median(lum) - lum.mean())

    log_lum = np.log(lum + 1e-4)
    f.update(_moments('log_lum', log_lum))
    for gamma in (0.45, 0.67, 1.5, 2.2):
        f[f'gamma_{str(gamma).replace(".", "_")}_mean'] = float(np.power(np.clip(lum, 0, 1), gamma).mean())

    return f


def _zone_features(lum: np.ndarray, n: int, prefix: str) -> dict[str, float]:
    h, w = lum.shape
    y_edges = np.linspace(0, h, n + 1, dtype=int)
    x_edges = np.linspace(0, w, n + 1, dtype=int)
    out: dict[str, float] = {}
    idx = 0
    for yi in range(n):
        for xi in range(n):
            zone = lum[y_edges[yi]:y_edges[yi + 1], x_edges[xi]:x_edges[xi + 1]]
            out[f'{prefix}_mean_{idx:02d}'] = float(zone.mean())
            out[f'{prefix}_std_{idx:02d}'] = float(zone.std())
            idx += 1
    return out


def _haar_energies(gray: np.ndarray) -> tuple[float, float, float, float]:
    h, w = gray.shape
    h -= h % 2
    w -= w % 2
    x = gray[:h, :w]
    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]
    ll = (a + b + c + d) / 4.0
    lh = (a - b + c - d) / 4.0
    hl = (a + b - c - d) / 4.0
    hh = (a - b - c + d) / 4.0
    return tuple(float(np.mean(v * v)) for v in (ll, lh, hl, hh))


def extract_spatial_features(path: Path | str) -> dict[str, float]:
    rgb = _load_rgb(path)
    lum = _luminance(rgb).astype(np.float32)
    if min(lum.shape) < 32:
        lum = cv2.resize(lum, (64, 64), interpolation=cv2.INTER_AREA)

    f: dict[str, float] = {}
    f.update(_zone_features(lum, 4, 'zone4'))
    f.update(_zone_features(lum, 8, 'zone8'))

    h, w = lum.shape
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    center = lum[y0:y1, x0:x1]
    edge_mask = np.ones_like(lum, dtype=bool)
    edge_mask[y0:y1, x0:x1] = False
    edge = lum[edge_mask]
    f['center_mean'] = float(center.mean())
    f['edge_mean'] = float(edge.mean())
    f['center_minus_edge'] = float(center.mean() - edge.mean())

    sobel_x = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    f.update(_moments('grad', grad))
    lap = cv2.Laplacian(lum, cv2.CV_32F)
    f['laplacian_var'] = float(lap.var())
    f['laplacian_abs_mean'] = float(np.abs(lap).mean())

    for sigma in (1.0, 2.0, 4.0, 8.0):
        blur = cv2.GaussianBlur(lum, (0, 0), sigmaX=sigma)
        diff = lum - blur
        f[f'dog_sigma_{int(sigma)}_abs_mean'] = float(np.abs(diff).mean())
        f[f'dog_sigma_{int(sigma)}_std'] = float(diff.std())

    ll, h_energy, v_energy, d_energy = _haar_energies(lum)
    f['wavelet_ll_energy'] = ll
    f['wavelet_h_energy'] = h_energy
    f['wavelet_v_energy'] = v_energy
    f['wavelet_d_energy'] = d_energy

    blur = cv2.GaussianBlur(lum, (0, 0), sigmaX=15.0)
    retinex = np.log(lum + 1e-3) - np.log(blur + 1e-3)
    f.update(_moments('retinex', retinex))
    f['retinex_abs_mean'] = float(np.abs(retinex).mean())
    return f


def extract_feature_table(paths: Iterable[Path | str], mode: str = 'all') -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for path in paths:
        if mode == 'global':
            row = extract_global_features(path)
        elif mode == 'spatial':
            row = extract_spatial_features(path)
        elif mode == 'all':
            row = extract_global_features(path)
            row.update(extract_spatial_features(path))
        else:
            raise ValueError("mode must be one of: global, spatial, all")
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = frame.reindex(sorted(frame.columns), axis=1)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
