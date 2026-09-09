from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _occupancy_fit(occupied: np.ndarray, scale: float = 1.0, gamma: float = 1.0) -> float:
    """Jaccard similarity to the set of 8-bit levels reachable by a deterministic tone curve."""
    x = np.arange(256, dtype=np.float64) / 255.0
    y = np.clip(np.rint(scale * np.power(x, gamma) * 255.0), 0, 255).astype(np.int32)
    reachable = np.zeros(256, dtype=bool)
    reachable[np.unique(y)] = True
    observed = np.asarray(occupied, dtype=bool)
    union = np.logical_or(observed, reachable).sum()
    return float(np.logical_and(observed, reachable).sum() / max(union, 1))


def _channel_features(prefix: str, x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.uint8).ravel()
    hist = np.bincount(x, minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    occ = hist > 0
    levels = np.flatnonzero(occ)
    out: dict[str, float] = {}
    out[f"{prefix}_unique"] = float(len(levels))
    out[f"{prefix}_zero_bins"] = float((~occ).sum())
    out[f"{prefix}_zero_mid_bins"] = float((~occ[16:240]).sum())
    out[f"{prefix}_min_level"] = float(levels[0] if len(levels) else 0)
    out[f"{prefix}_max_level"] = float(levels[-1] if len(levels) else 0)
    out[f"{prefix}_clip0"] = float(p[0])
    out[f"{prefix}_clip255"] = float(p[255])
    out[f"{prefix}_clip_lo8"] = float(p[:8].sum())
    out[f"{prefix}_clip_hi8"] = float(p[-8:].sum())
    nz = p[p > 0]
    out[f"{prefix}_entropy"] = float(-(nz * np.log2(nz)).sum()) if len(nz) else 0.0
    out[f"{prefix}_hist_roughness"] = float(np.abs(np.diff(p)).mean())
    out[f"{prefix}_hist_second_diff"] = float(np.abs(np.diff(p, n=2)).mean())
    out[f"{prefix}_peak_prob"] = float(p.max())
    if len(levels) > 1:
        gaps = np.diff(levels)
        out[f"{prefix}_gap_mean"] = float(gaps.mean())
        out[f"{prefix}_gap_std"] = float(gaps.std())
        out[f"{prefix}_gap_max"] = float(gaps.max())
        for gap in (1, 2, 3, 4, 5, 6, 7, 8):
            out[f"{prefix}_gap_eq{gap}"] = float(np.mean(gaps == gap))
    else:
        for name in ("gap_mean", "gap_std", "gap_max"):
            out[f"{prefix}_{name}"] = 0.0
        for gap in range(1, 9):
            out[f"{prefix}_gap_eq{gap}"] = 0.0

    centered = occ.astype(np.float64) - occ.mean()
    sp = np.abs(np.fft.rfft(centered))
    for k in (1, 2, 3, 4, 5, 6, 8, 16, 32, 64):
        if k < len(sp):
            out[f"{prefix}_occ_fft{k}"] = float(sp[k] / 256.0)
    out[f"{prefix}_occ_fft_peak"] = float(sp[1:].max() / 256.0) if len(sp) > 1 else 0.0

    for mod in (2, 3, 4, 5, 6, 7, 8, 16, 32):
        counts = np.bincount(x % mod, minlength=mod).astype(np.float64)
        counts /= max(counts.sum(), 1.0)
        uniform = 1.0 / mod
        out[f"{prefix}_mod{mod}_maxdev"] = float(np.max(np.abs(counts - uniform)))
        out[f"{prefix}_mod{mod}_l1"] = float(np.abs(counts - uniform).sum())
        q = counts[counts > 0]
        out[f"{prefix}_mod{mod}_entropy"] = float(-(q * np.log2(q)).sum())
    for bit in range(8):
        out[f"{prefix}_bit{bit}"] = float(np.mean((x >> bit) & 1))

    # Fit common linear exposure multipliers and gamma curves to occupied-level fingerprints.
    fits = []
    for scale in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0):
        score = _occupancy_fit(occ, scale=scale, gamma=1.0)
        out[f"{prefix}_scale_fit_{str(scale).replace('.', '_')}"] = score
        fits.append((score, scale))
    for gamma in (0.45, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2):
        score = _occupancy_fit(occ, scale=1.0, gamma=gamma)
        out[f"{prefix}_gamma_fit_{str(gamma).replace('.', '_')}"] = score
    out[f"{prefix}_best_scale"] = float(max(fits)[1])
    out[f"{prefix}_best_scale_fit"] = float(max(fits)[0])
    return out


def extract_raw_quant_features(path: Path | str) -> dict[str, float]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    r, g, b = np.moveaxis(rgb, -1, 0)
    # Use integer luma without interpolation so tone-curve fingerprints are preserved.
    lum = ((54 * r.astype(np.uint16) + 183 * g.astype(np.uint16) + 19 * b.astype(np.uint16) + 128) >> 8).astype(np.uint8)
    vmax = np.max(rgb, axis=2).astype(np.uint8)
    vmin = np.min(rgb, axis=2).astype(np.uint8)

    out: dict[str, float] = {}
    for name, values in (("rq_r", r), ("rq_g", g), ("rq_b", b), ("rq_lum", lum), ("rq_max", vmax), ("rq_min", vmin)):
        out.update(_channel_features(name, values))

    # Difference residues can reveal scaling/quantization even when marginal histograms are broad.
    for cname, values in (("r", r), ("g", g), ("b", b), ("lum", lum)):
        dx = np.abs(np.diff(values.astype(np.int16), axis=1)).astype(np.uint16).ravel()
        dy = np.abs(np.diff(values.astype(np.int16), axis=0)).astype(np.uint16).ravel()
        d = np.concatenate([dx, dy])
        d = d[d <= 255].astype(np.uint8)
        hist = np.bincount(d, minlength=256).astype(float)
        hist /= max(hist.sum(), 1.0)
        out[f"rq_{cname}_diff_zero"] = float(hist[0])
        out[f"rq_{cname}_diff_one"] = float(hist[1])
        out[f"rq_{cname}_diff_two"] = float(hist[2])
        for mod in (2, 3, 4, 5, 8, 16):
            c = np.bincount(d % mod, minlength=mod).astype(float)
            c /= max(c.sum(), 1.0)
            out[f"rq_{cname}_diff_mod{mod}_maxdev"] = float(np.max(np.abs(c - 1.0 / mod)))

    return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in sorted(out.items())}
