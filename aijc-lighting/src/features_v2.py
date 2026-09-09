from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image

EPS = 1e-6


def _load(path: Path | str, max_side: int = 256) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        nh = max(32, int(round(h * scale)))
        nw = max(32, int(round(w * scale)))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return rgb


def _stats(
    out: dict[str, float],
    prefix: str,
    values: np.ndarray,
    percentiles=(1, 5, 10, 25, 50, 75, 90, 95, 99),
) -> None:
    x = np.asarray(values, dtype=np.float32).ravel()
    mean = float(x.mean())
    std = float(x.std())
    out[f"{prefix}_mean"] = mean
    out[f"{prefix}_std"] = std
    for q in percentiles:
        out[f"{prefix}_p{q:02d}"] = float(np.percentile(x, q))
    if std > 1e-8:
        z = (x - mean) / (std + EPS)
        out[f"{prefix}_skew"] = float(np.mean(z**3))
        out[f"{prefix}_kurt"] = float(np.mean(z**4) - 3.0)
    else:
        out[f"{prefix}_skew"] = 0.0
        out[f"{prefix}_kurt"] = 0.0


def _hist(
    out: dict[str, float],
    prefix: str,
    values: np.ndarray,
    bins: int = 16,
    lo: float = 0.0,
    hi: float = 1.0,
) -> None:
    hist, _ = np.histogram(np.asarray(values), bins=bins, range=(lo, hi))
    hist = hist.astype(float)
    hist /= max(hist.sum(), 1.0)
    for i, value in enumerate(hist):
        out[f"{prefix}_{i:02d}"] = float(value)
    nz = hist[hist > 0]
    out[f"{prefix}_entropy"] = float(-(nz * np.log2(nz)).sum()) if len(nz) else 0.0


def _quantization_features(out: dict[str, float], prefix: str, values: np.ndarray) -> None:
    """Fingerprint deterministic brightness/gamma transforms in 8-bit output."""
    x = np.clip(np.asarray(values) * 255.0, 0, 255).round().astype(np.uint8).ravel()
    hist = np.bincount(x, minlength=256).astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    occupied = (hist > 0).astype(np.float64)
    unique = np.flatnonzero(hist > 0)

    out[f"{prefix}_unique_count"] = float(len(unique))
    out[f"{prefix}_zero_bins"] = float(np.sum(hist == 0))
    out[f"{prefix}_hist_roughness"] = float(np.abs(np.diff(prob)).mean())
    out[f"{prefix}_hist_peak"] = float(prob.max())
    if len(unique) > 1:
        gaps = np.diff(unique)
        out[f"{prefix}_gap_mean"] = float(gaps.mean())
        out[f"{prefix}_gap_max"] = float(gaps.max())
        out[f"{prefix}_gap_gt1"] = float(np.mean(gaps > 1))
    else:
        out[f"{prefix}_gap_mean"] = 0.0
        out[f"{prefix}_gap_max"] = 0.0
        out[f"{prefix}_gap_gt1"] = 0.0

    spectrum = np.abs(np.fft.rfft(occupied - occupied.mean()))
    out[f"{prefix}_occupancy_fft_peak"] = (
        float(spectrum[1:].max() / max(len(occupied), 1)) if len(spectrum) > 1 else 0.0
    )
    for mod in (2, 3, 4, 5, 6, 7, 8, 16):
        counts = np.bincount(x % mod, minlength=mod).astype(float)
        counts /= max(counts.sum(), 1.0)
        out[f"{prefix}_mod{mod}_maxdev"] = float(np.max(np.abs(counts - 1.0 / mod)))
        nz = counts[counts > 0]
        out[f"{prefix}_mod{mod}_entropy"] = float(-(nz * np.log2(nz)).sum())
    for bit in range(4):
        out[f"{prefix}_bit{bit}_one"] = float(np.mean((x >> bit) & 1))


def extract_v2_features(path: Path | str) -> dict[str, float]:
    rgb = _load(path)
    r, g, b = np.moveaxis(rgb, -1, 0)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    u8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[..., 0] / 179.0
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_l = lab[..., 0] / 255.0
    lab_a = (lab[..., 1] - 128.0) / 127.0
    lab_b = (lab[..., 2] - 128.0) / 127.0

    f: dict[str, float] = {}
    _stats(f, "v2_lum", lum)
    _hist(f, "v2_lum_hist", lum, 32)
    for threshold in (4, 8, 16, 24, 32, 48, 64, 80, 96):
        f[f"v2_shadow_frac_{threshold}"] = float((lum * 255 <= threshold).mean())
    for threshold in (160, 176, 192, 208, 224, 232, 240, 248, 252):
        f[f"v2_highlight_frac_{threshold}"] = float((lum * 255 >= threshold).mean())

    channels = [
        ("r", r), ("g", g), ("b", b), ("h", hue), ("s", sat), ("v", val),
        ("lab_l", lab_l), ("lab_a", lab_a), ("lab_b", lab_b),
    ]
    for name, values in channels:
        _stats(f, f"v2_{name}", values, (5, 25, 50, 75, 95))
        if name in {"r", "g", "b", "s", "v"}:
            _hist(f, f"v2_{name}_hist", values, 12)

    total = r + g + b + EPS
    for name, values in (("cr", r / total), ("cg", g / total), ("cb", b / total)):
        _stats(f, f"v2_{name}", values, (10, 50, 90))
    for name, values in (
        ("log_rg", np.log(r + 1e-3) - np.log(g + 1e-3)),
        ("log_bg", np.log(b + 1e-3) - np.log(g + 1e-3)),
        ("log_rb", np.log(r + 1e-3) - np.log(b + 1e-3)),
    ):
        _stats(f, f"v2_{name}", values, (10, 50, 90))

    means = np.array([r.mean(), g.mean(), b.mean()], float)
    maxima = np.array([r.max(), g.max(), b.max()], float)
    shades = np.array([(np.mean(c**6) + EPS) ** (1 / 6) for c in (r, g, b)])
    for name, vector in (("grayworld", means), ("maxrgb", maxima), ("sog6", shades)):
        vector = vector / (np.linalg.norm(vector) + EPS)
        for i, channel in enumerate("rgb"):
            f[f"v2_{name}_{channel}"] = float(vector[i])
        f[f"v2_{name}_neutral_dev"] = float(np.sum((vector - 1 / np.sqrt(3)) ** 2))

    h, w = lum.shape
    y_edges = np.linspace(0, h, 5, dtype=int)
    x_edges = np.linspace(0, w, 5, dtype=int)
    cell_means = []
    for yi in range(4):
        for xi in range(4):
            zone = lum[y_edges[yi]:y_edges[yi + 1], x_edges[xi]:x_edges[xi + 1]]
            index = yi * 4 + xi
            mean = float(zone.mean())
            cell_means.append(mean)
            f[f"v2_cell_mean_{index:02d}"] = mean
            f[f"v2_cell_std_{index:02d}"] = float(zone.std())
            f[f"v2_cell_p10_{index:02d}"] = float(np.percentile(zone, 10))
            f[f"v2_cell_p90_{index:02d}"] = float(np.percentile(zone, 90))
            f[f"v2_cell_shadow_{index:02d}"] = float((zone < 0.125).mean())
            f[f"v2_cell_high_{index:02d}"] = float((zone > 0.875).mean())
    cell_means = np.asarray(cell_means)
    f["v2_cell_mean_std"] = float(cell_means.std())
    f["v2_cell_mean_range"] = float(cell_means.max() - cell_means.min())
    f["v2_top_bottom_diff"] = float(lum[:h // 2].mean() - lum[h // 2:].mean())
    f["v2_left_right_diff"] = float(lum[:, :w // 2].mean() - lum[:, w // 2:].mean())

    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    _stats(f, "v2_grad", grad, (50, 75, 90, 95, 99))
    local = cv2.GaussianBlur(lum, (0, 0), 3)
    _stats(f, "v2_norm_grad", grad / (local + 0.03), (50, 75, 90, 95, 99))
    lap = np.abs(cv2.Laplacian(lum, cv2.CV_32F))
    _stats(f, "v2_lap_abs", lap, (50, 90, 99))

    for kernel in (5, 11, 21, 41):
        local_mean = cv2.blur(lum, (kernel, kernel))
        local_sq = cv2.blur(lum * lum, (kernel, kernel))
        rms = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0))
        _stats(f, f"v2_rms{kernel}", rms, (10, 50, 90, 99))
        f[f"v2_weber{kernel}"] = float(np.mean(rms / (local_mean + 0.03)))

    highpass = lum - cv2.GaussianBlur(lum, (0, 0), 1.0)
    for name, mask in (
        ("dark", lum < 0.2),
        ("mid", (lum >= 0.2) & (lum < 0.7)),
        ("bright", lum >= 0.7),
    ):
        values = highpass[mask]
        f[f"v2_noise_{name}_std"] = float(values.std()) if values.size else 0.0
        f[f"v2_noise_{name}_abs"] = float(np.abs(values).mean()) if values.size else 0.0
        f[f"v2_region_{name}_frac"] = float(mask.mean())

    log_lum = np.log(lum + 1e-3)
    for sigma in (3, 8, 15, 30):
        illum = cv2.GaussianBlur(lum, (0, 0), sigmaX=sigma)
        retinex = log_lum - np.log(illum + 1e-3)
        _stats(f, f"v2_illum_s{sigma}", illum, (5, 50, 95))
        _stats(f, f"v2_ret_s{sigma}", retinex, (5, 50, 95))

    minimum = np.min(rgb, axis=2)
    maximum = np.max(rgb, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dark_channel = cv2.erode(minimum, kernel)
    bright_channel = cv2.dilate(maximum, kernel)
    _stats(f, "v2_dark_channel", dark_channel, (1, 10, 50, 90, 99))
    _stats(f, "v2_bright_channel", bright_channel, (1, 10, 50, 90, 99))

    gamma_means: dict[float, float] = {}
    for gamma in (0.35, 0.5, 0.7, 1.3, 1.7, 2.2, 3.0):
        gamma_means[gamma] = float(np.mean(np.power(np.clip(lum, 0, 1), gamma)))
        f[f"v2_gamma_{str(gamma).replace('.', '_')}"] = gamma_means[gamma]
    f["v2_gamma_gap_lo_hi"] = gamma_means[0.5] - gamma_means[2.2]
    f["v2_gamma_gap_extreme"] = gamma_means[0.35] - gamma_means[3.0]

    small = cv2.resize(lum, (64, 64), interpolation=cv2.INTER_AREA)
    normalized = (small - small.mean()) / (small.std() + EPS)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(normalized))) ** 2
    yy, xx = np.indices(spectrum.shape)
    radius = np.sqrt((yy - 31.5) ** 2 + (xx - 31.5) ** 2)
    total_energy = float(spectrum.sum() + EPS)
    for i, (low, high) in enumerate(((0, 4), (4, 8), (8, 16), (16, 24), (24, 46))):
        f[f"v2_fft_band_{i}"] = float(spectrum[(radius >= low) & (radius < high)].sum() / total_energy)

    dct = cv2.dct(normalized.astype(np.float32))
    block = dct[:6, :6].copy()
    block[0, 0] = 0
    scale = float(np.linalg.norm(block) + EPS)
    for i, value in enumerate((block / scale).ravel()):
        f[f"v2_dct_{i:02d}"] = float(value)

    # Exposure generators often leave strong 8-bit occupancy / residue patterns.
    for name, values in (("r", r), ("g", g), ("b", b), ("lum", lum), ("v", val)):
        _quantization_features(f, f"v2_q_{name}", values)

    p = np.percentile(lum, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    f["v2_dr_90_10"] = float(p[6] - p[2])
    f["v2_dr_99_01"] = float(p[8] - p[0])
    f["v2_iqr"] = float(p[5] - p[3])
    f["v2_low_high_mass_ratio"] = float(((lum < 0.25).mean() + EPS) / ((lum > 0.75).mean() + EPS))

    return {
        key: (0.0 if not np.isfinite(value) else float(value))
        for key, value in sorted(f.items())
    }


def extract_v2_feature_frame(paths: Iterable[Path | str]) -> pd.DataFrame:
    frame = pd.DataFrame([extract_v2_features(path) for path in paths])
    return frame.reindex(sorted(frame.columns), axis=1).replace([np.inf, -np.inf], 0).fillna(0.0)
