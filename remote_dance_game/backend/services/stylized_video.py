from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


POSE_CONNECTIONS = [
    (0, 7), (0, 8), (7, 9), (8, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]
LEFT_RIGHT_SWAP_PAIRS = [
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


@dataclass(frozen=True)
class Theme:
    name: str
    primary: Tuple[int, int, int]      # RGB
    secondary: Tuple[int, int, int]
    accent: Tuple[int, int, int]
    deep: Tuple[int, int, int]
    motif: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "primary": list(self.primary),
            "secondary": list(self.secondary),
            "accent": list(self.accent),
            "deep": list(self.deep),
            "motif": self.motif,
        }


THEMES: Sequence[Theme] = (
    Theme("Electric Bloom", (37, 244, 255), (255, 47, 181), (255, 236, 94), (8, 5, 35), "rings"),
    Theme("Velvet Nova", (152, 88, 255), (255, 78, 141), (83, 255, 207), (14, 6, 38), "ribbons"),
    Theme("Solar Arcade", (255, 131, 46), (255, 51, 153), (70, 241, 255), (28, 7, 30), "sunset"),
    Theme("Laser Garden", (74, 255, 133), (49, 203, 255), (243, 89, 255), (4, 24, 26), "grid"),
    Theme("Midnight Pop", (77, 122, 255), (255, 77, 222), (255, 244, 117), (5, 8, 33), "stars"),
)


def choose_theme(title: str, tempo: float = 120.0) -> Theme:
    seed = zlib.crc32((title or "dance").encode("utf-8")) & 0xFFFFFFFF
    tempo_bucket = int(max(0.0, tempo) // 12)
    return THEMES[(seed + tempo_bucket) % len(THEMES)]


def _bgr(rgb: Sequence[int]) -> Tuple[int, int, int]:
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _visible_landmarks(frame: Dict[str, Any], mirror_mode: bool = False) -> List[Dict[str, float]]:
    points = [dict(x) for x in frame.get("landmarks", [])[:33]]
    if len(points) < 33:
        points += [{"x": 0.0, "y": 0.0, "v": 0.0}] * (33 - len(points))
    if mirror_mode:
        for p in points:
            p["x"] = 1.0 - float(p.get("x", 0.0))
        for i, j in LEFT_RIGHT_SWAP_PAIRS:
            points[i], points[j] = points[j], points[i]
    return points


def _pose_bbox(frame: Dict[str, Any], width: int, height: int, mirror_mode: bool) -> Optional[Tuple[int, int, int, int]]:
    pts = _visible_landmarks(frame, mirror_mode)
    xy = [
        (float(p.get("x", 0.0)) * width, float(p.get("y", 0.0)) * height)
        for p in pts
        if float(p.get("v", 0.0)) >= 0.25 and 0 <= float(p.get("x", 0.0)) <= 1 and 0 <= float(p.get("y", 0.0)) <= 1
    ]
    if len(xy) < 6:
        return None
    xs, ys = zip(*xy)
    pad_x = max(18, int((max(xs) - min(xs)) * 0.16))
    pad_y = max(18, int((max(ys) - min(ys)) * 0.12))
    return (
        max(0, int(min(xs)) - pad_x), max(0, int(min(ys)) - pad_y),
        min(width - 1, int(max(xs)) + pad_x), min(height - 1, int(max(ys)) + pad_y),
    )


def _pose_mask(frame: Dict[str, Any], width: int, height: int, mirror_mode: bool = False) -> np.ndarray:
    pts = _visible_landmarks(frame, mirror_mode)
    mask = np.zeros((height, width), np.uint8)
    visible = [float(p.get("v", 0.0)) >= 0.22 for p in pts]

    shoulder_px = 0.12 * width
    if visible[11] and visible[12]:
        shoulder_px = max(22.0, abs((float(pts[11]["x"]) - float(pts[12]["x"])) * width))
    thick = max(10, int(shoulder_px * 0.34))

    def point(i: int) -> Tuple[int, int]:
        return int(float(pts[i].get("x", 0.0)) * width), int(float(pts[i].get("y", 0.0)) * height)

    for a, b in POSE_CONNECTIONS:
        if visible[a] and visible[b]:
            cv2.line(mask, point(a), point(b), 255, thick, cv2.LINE_AA)
    for i in range(33):
        if visible[i]:
            radius = thick // 2
            if i in {0, 7, 8, 9, 10}:
                radius = int(thick * 0.95)
            elif i in {15, 16, 27, 28, 31, 32}:
                radius = int(thick * 0.65)
            cv2.circle(mask, point(i), max(5, radius), 255, -1, cv2.LINE_AA)

    torso_ids = [11, 12, 24, 23]
    if all(visible[i] for i in torso_ids):
        polygon = np.array([point(i) for i in torso_ids], np.int32)
        cv2.fillConvexPoly(mask, polygon, 255, cv2.LINE_AA)

    # Expand the core into a soft but complete body-shaped fallback matte.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(9, thick // 2 * 2 + 1),) * 2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask.astype(np.float32) / 255.0


class PersonSegmenter:
    """YOLO person instance mask with pose-guided selection and safe fallback.

    The expensive model is intentionally optional. The renderer remains functional
    offline with a pose-derived matte when ultralytics/model weights are unavailable.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model = None
        self.device: Any = None
        self.backend = "pose_fallback"
        self.model_name = model_name or os.environ.get("DANCE_SEGMENTATION_MODEL", "yolo11s-seg.pt")
        if os.environ.get("DANCE_DISABLE_YOLO", "").lower() in {"1", "true", "yes"}:
            return
        try:
            from ultralytics import YOLO
            import torch
            self.device = 0 if torch.cuda.is_available() else "cpu"
            self.model = YOLO(self.model_name)
            self.backend = f"yolo:{self.model_name}"
        except Exception as exc:
            print(f"[stylizer] YOLO segmentation unavailable, using pose matte: {exc}")
            self.model = None

    def segment(self, frame: np.ndarray, pose_frame: Dict[str, Any], mirror_mode: bool = False) -> Tuple[np.ndarray, str]:
        h, w = frame.shape[:2]
        fallback = _pose_mask(pose_frame, w, h, mirror_mode)
        if self.model is None:
            return fallback, "pose_fallback"

        try:
            result = self.model.predict(
                source=frame,
                classes=[0],
                conf=0.22,
                imgsz=640,
                verbose=False,
                device=self.device,
            )[0]
            if result.masks is None or result.boxes is None or len(result.boxes) == 0:
                return fallback, "pose_fallback"

            masks = result.masks.data.detach().cpu().numpy()
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            pose_box = _pose_bbox(pose_frame, w, h, mirror_mode)
            if pose_box is not None:
                px1, py1, px2, py2 = pose_box
                pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            else:
                pcx, pcy = w / 2.0, h / 2.0

            best_idx = 0
            best_score = -1e9
            diag = max(1.0, math.hypot(w, h))
            for i, (box, conf) in enumerate(zip(boxes, confs)):
                x1, y1, x2, y2 = [float(v) for v in box]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dist = math.hypot(cx - pcx, cy - pcy) / diag
                contains = 1.0 if x1 <= pcx <= x2 and y1 <= pcy <= y2 else 0.0
                area = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, w * h)
                score = float(conf) * 1.4 + contains * 0.65 - dist * 1.6 + min(area, 0.7) * 0.18
                if score > best_score:
                    best_score = score
                    best_idx = i

            mask = masks[best_idx].astype(np.float32)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = _clamp01(mask)
            # Pose core protects hands/feet when segmentation under-fills fine limbs.
            mask = np.maximum(mask, fallback * 0.78)
            return mask, self.backend
        except Exception as exc:
            print(f"[stylizer] segmentation frame failed, fallback used: {exc}")
            return fallback, "pose_fallback"


def _nearest_pose(frames: Sequence[Dict[str, Any]], t_ms: int, hint: int) -> Tuple[Dict[str, Any], int]:
    if not frames:
        return {"t_ms": t_ms, "landmarks": []}, 0
    i = min(max(hint, 0), len(frames) - 1)
    while i + 1 < len(frames) and int(frames[i + 1].get("t_ms", 0)) <= t_ms:
        i += 1
    while i > 0 and int(frames[i].get("t_ms", 0)) > t_ms:
        i -= 1
    if i + 1 < len(frames):
        a = abs(int(frames[i].get("t_ms", 0)) - t_ms)
        b = abs(int(frames[i + 1].get("t_ms", 0)) - t_ms)
        if b < a:
            i += 1
    return frames[i], i


def _beat_strength(t_ms: float, beat_ms: Sequence[float], strong_ms: Sequence[float]) -> float:
    best = 0.0
    for beats, multiplier in ((beat_ms, 0.75), (strong_ms, 1.0)):
        if not beats:
            continue
        # Timelines are sorted; binary-search nearest point.
        arr = np.asarray(beats, dtype=np.float32)
        idx = int(np.searchsorted(arr, t_ms))
        for j in (idx - 1, idx):
            if 0 <= j < len(arr):
                d = abs(float(arr[j]) - t_ms)
                if d < 230:
                    best = max(best, multiplier * math.exp(-((d / 105.0) ** 2)))
    return min(1.0, best)


def _background(width: int, height: int, t: float, beat: float, theme: Theme, seed: int) -> np.ndarray:
    # Render procedural art at quarter resolution then upscale. It remains smooth,
    # while keeping 1080p export practical on CPU.
    sw, sh = max(320, width // 4), max(180, height // 4)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    xn = xx / max(1, sw - 1)
    yn = yy / max(1, sh - 1)

    c0 = np.array(_bgr(theme.deep), np.float32)
    c1 = np.array(_bgr(theme.primary), np.float32)
    c2 = np.array(_bgr(theme.secondary), np.float32)
    wave = 0.5 + 0.5 * np.sin((xn * 2.1 + yn * 1.35 + t * 0.065) * math.tau)
    radial = np.sqrt((xn - (0.46 + 0.08 * math.sin(t * 0.31))) ** 2 + (yn - 0.48) ** 2)
    glow = np.clip(1.0 - radial * 1.65, 0, 1)
    mix1 = (0.18 + 0.23 * wave + 0.24 * glow + 0.12 * beat)[..., None]
    mix2 = (0.06 + 0.18 * (1 - wave) + 0.16 * beat)[..., None]
    bg = c0 + (c1 - c0) * mix1 + (c2 - c0) * mix2
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    rng = np.random.default_rng(seed)
    primary = _bgr(theme.primary)
    secondary = _bgr(theme.secondary)
    accent = _bgr(theme.accent)
    phase = t * 32.0

    if theme.motif in {"rings", "stars"}:
        center = (int(sw * (0.5 + 0.07 * math.sin(t * 0.23))), int(sh * 0.52))
        for k in range(5):
            radius = int((0.16 + k * 0.13 + (t * 0.035) % 0.13) * max(sw, sh))
            col = primary if k % 2 == 0 else secondary
            cv2.circle(bg, center, radius, col, max(1, 1 + int(beat * 2)), cv2.LINE_AA)
    elif theme.motif == "grid":
        horizon = int(sh * 0.57)
        for i in range(-8, 9):
            x = int(sw / 2 + i * sw * 0.085)
            cv2.line(bg, (sw // 2, horizon), (x, sh), primary if i % 2 else secondary, 1, cv2.LINE_AA)
        offset = int((phase * 0.9) % 22)
        for y in range(horizon + offset, sh, 22):
            cv2.line(bg, (0, y), (sw, y), accent, 1, cv2.LINE_AA)
    elif theme.motif == "sunset":
        center = (int(sw * 0.5), int(sh * 0.47))
        cv2.circle(bg, center, int(sh * (0.19 + 0.025 * beat)), accent, -1, cv2.LINE_AA)
        for y in range(center[1] - int(sh * 0.18), center[1] + int(sh * 0.18), 8):
            cv2.line(bg, (center[0] - int(sw * 0.18), y), (center[0] + int(sw * 0.18), y), _bgr(theme.deep), 2)
    else:  # ribbons
        for k in range(4):
            pts = []
            for x in range(-20, sw + 20, 12):
                y = int(sh * (0.26 + k * 0.16) + math.sin(x * 0.025 + t * (0.8 + k * 0.08) + k) * sh * 0.07)
                pts.append((x, y))
            cv2.polylines(bg, [np.array(pts, np.int32)], False, primary if k % 2 == 0 else secondary, 2 + int(beat * 2), cv2.LINE_AA)

    # Fixed particles move deterministically with time rather than flickering.
    for i in range(34):
        bx, by = rng.random(), rng.random()
        speed = 0.018 + rng.random() * 0.042
        x = int(((bx + t * speed) % 1.08 - 0.04) * sw)
        y = int(((by + math.sin(t * 0.3 + i) * 0.02) % 1.0) * sh)
        r = 1 + int(rng.random() * 3 + beat * 1.5)
        cv2.circle(bg, (x, y), r, accent if i % 3 == 0 else primary, -1, cv2.LINE_AA)

    bg = cv2.GaussianBlur(bg, (0, 0), 0.6)
    return cv2.resize(bg, (width, height), interpolation=cv2.INTER_CUBIC)


def _stylize_person(frame: np.ndarray, mask: np.ndarray, pose_frame: Dict[str, Any], theme: Theme, beat: float, mirror_mode: bool) -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.25)
    soft = _clamp01(soft)

    # Bold posterized texture retains clothing folds/facial structure while replacing
    # photographic colour with the game palette.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = np.clip((gray - 0.18) * 1.45, 0, 1)
    levels = np.round(gray * 5.0) / 5.0
    c1 = np.array(_bgr(theme.primary), np.float32)
    c2 = np.array(_bgr(theme.secondary), np.float32)
    accent = np.array(_bgr(theme.accent), np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    tint = c1[None, None, :] * (1.0 - y) + c2[None, None, :] * y
    tint = np.broadcast_to(tint, (h, w, 3)).copy()
    lum = (0.34 + 0.78 * levels)[..., None]
    stylized = tint * lum
    # Keep the dancer visually stable. Beat-reactive energy belongs behind the
    # coach, never across the body, so fast gestures remain easy to read.
    stylized = np.clip(stylized * (1.0 + 0.025 * beat), 0, 255)

    # White face + hands, Just-Dance-like visual anchors that make fast gestures readable.
    pts = _visible_landmarks(pose_frame, mirror_mode)
    anchor_mask = np.zeros((h, w), np.uint8)
    shoulder = 0.09 * w
    if float(pts[11].get("v", 0)) > .25 and float(pts[12].get("v", 0)) > .25:
        shoulder = max(18, abs(float(pts[11]["x"]) - float(pts[12]["x"])) * w)
    for idx, scale in ((0, .44), (15, .24), (16, .24), (19, .18), (20, .18)):
        p = pts[idx]
        if float(p.get("v", 0)) > .22:
            cv2.circle(anchor_mask, (int(float(p["x"]) * w), int(float(p["y"]) * h)), max(5, int(shoulder * scale)), 255, -1, cv2.LINE_AA)
    anchor = (anchor_mask.astype(np.float32) / 255.0 * soft)[..., None]
    stylized = stylized * (1.0 - anchor * 0.9) + np.array([248, 252, 255], np.float32) * anchor * 0.9

    # Edge and glow are derived from the matte, so they remain stable even when the
    # source background has high-frequency detail.
    hard = (soft > 0.46).astype(np.uint8) * 255
    edge = cv2.morphologyEx(hard, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)).astype(np.float32) / 255.0
    edge = cv2.GaussianBlur(edge, (0, 0), 0.7)
    outline = edge[..., None]
    stylized = stylized * (1.0 - outline * 0.82) + accent * outline * 0.82
    return np.clip(stylized, 0, 255).astype(np.uint8), soft


def _draw_stage_shadow(bg: np.ndarray, pose_frame: Dict[str, Any], mask: np.ndarray, theme: Theme, mirror_mode: bool) -> None:
    h, w = bg.shape[:2]
    pts = _visible_landmarks(pose_frame, mirror_mode)
    feet = []
    for idx in (27, 28, 31, 32):
        p = pts[idx]
        if float(p.get("v", 0)) > .2:
            feet.append((float(p["x"]) * w, float(p["y"]) * h))
    if not feet:
        ys, xs = np.where(mask > .42)
        if len(xs) == 0:
            return
        cx, cy = float(xs.mean()), float(np.percentile(ys, 98))
        span = max(40.0, float(np.percentile(xs, 90) - np.percentile(xs, 10)))
    else:
        cx = sum(p[0] for p in feet) / len(feet)
        cy = max(p[1] for p in feet)
        span = max(40.0, max(p[0] for p in feet) - min(p[0] for p in feet) + w * .09)
    shadow = np.zeros((h, w), np.uint8)
    cv2.ellipse(shadow, (int(cx), min(h - 5, int(cy + h * .018))), (int(span * .62), max(8, int(h * .018))), 0, 0, 360, 220, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(shadow, (0, 0), max(5, h * .012)).astype(np.float32) / 255.0
    col = np.array(_bgr(theme.primary), np.float32)
    bg[:] = np.clip(bg.astype(np.float32) + glow[..., None] * col * .42, 0, 255).astype(np.uint8)


def _compose(frame: np.ndarray, mask: np.ndarray, pose_frame: Dict[str, Any], theme: Theme, t: float, beat: float, seed: int, mirror_mode: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    bg = _background(w, h, t, beat, theme, seed)
    _draw_stage_shadow(bg, pose_frame, mask, theme, mirror_mode)
    dancer, soft = _stylize_person(frame, mask, pose_frame, theme, beat, mirror_mode)

    # A restrained halo is painted into the BACKGROUND first. The dancer is
    # composited last and is never covered by ribbons, bloom or beat flashes.
    glow = cv2.GaussianBlur(soft, (0, 0), max(8, w * 0.012))
    ring = np.clip(glow - soft * .70, 0, 1)[..., None]
    glow_col = np.array(_bgr(theme.primary), np.float32)
    out = np.clip(bg.astype(np.float32) + ring * glow_col * (0.28 + 0.12 * beat), 0, 255)
    alpha = np.clip(soft[..., None] * 1.08, 0, 1)
    out = out * (1.0 - alpha) + dancer.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _mux_audio(video_only: str, audio_path: Optional[str], output_path: str) -> None:
    ffmpeg = os.environ.get("FFMPEG_BIN", "ffmpeg")
    cmd = [ffmpeg, "-y", "-i", video_only]
    if audio_path and os.path.exists(audio_path):
        cmd += ["-i", audio_path, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", os.environ.get("DANCE_RENDER_PRESET", "medium"), "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "ffmpeg mux failed")[-3000:])


def render_game_video(
    input_path: str,
    output_path: str,
    pose_data: Dict[str, Any],
    timing_data: Optional[Dict[str, Any]] = None,
    audio_path: Optional[str] = None,
    title: str = "Dance",
    poster_path: Optional[str] = None,
    mirror_mode: bool = False,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    segmenter: Optional[PersonSegmenter] = None,
) -> Dict[str, Any]:
    """Compile source footage into a self-contained TV-friendly gameplay video."""
    if not os.path.isfile(input_path):
        raise ValueError(f"Video not found: {input_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if poster_path:
        os.makedirs(os.path.dirname(poster_path), exist_ok=True)

    timing = timing_data or {}
    tempo = float(timing.get("tempo", 120.0) or 120.0)
    beats = [float(x) for x in timing.get("beat_ms", [])]
    strong = [float(x) for x in timing.get("strong_beat_ms", [])]
    theme = choose_theme(title, tempo)
    seed = zlib.crc32((title + theme.name).encode("utf-8")) & 0xFFFFFFFF

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_src / max(src_fps, 1.0)
    total_out = max(1, int(round(duration_s * fps)))
    frames = pose_data.get("frames", []) or []

    seg = segmenter or PersonSegmenter()
    backends: Dict[str, int] = {}
    prev_mask: Optional[np.ndarray] = None
    pose_hint = 0
    poster_frame: Optional[np.ndarray] = None
    poster_target = int(total_out * .18)

    with tempfile.TemporaryDirectory(prefix="dance_render_") as td:
        silent_path = os.path.join(td, "stylized_silent.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(silent_path, fourcc, float(fps), (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not create temporary gameplay video")

        last_src_idx = -1
        src_frame: Optional[np.ndarray] = None
        for out_idx in range(total_out):
            t = out_idx / float(fps)
            target_src = min(total_src - 1, int(round(t * src_fps)))
            if target_src != last_src_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_src)
                ok, read = cap.read()
                if not ok:
                    break
                src_frame = read
                last_src_idx = target_src
            if src_frame is None:
                break
            frame = cv2.resize(src_frame, (width, height), interpolation=cv2.INTER_AREA)
            pose_frame, pose_hint = _nearest_pose(frames, int(t * 1000), pose_hint)
            current, backend = seg.segment(frame, pose_frame, mirror_mode=mirror_mode)
            backends[backend] = backends.get(backend, 0) + 1

            # Temporal matte smoothing. Fast motion still follows current frame because
            # current has higher weight; isolated one-frame segmentation holes disappear.
            if prev_mask is not None:
                current = np.maximum(current * .78 + prev_mask * .22, current * .88)
            current = cv2.morphologyEx((current * 255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(np.float32) / 255.0
            prev_mask = current

            beat = _beat_strength(t * 1000.0, beats, strong)
            out = _compose(frame, current, pose_frame, theme, t, beat, seed, mirror_mode)
            writer.write(out)
            if poster_frame is None and out_idx >= poster_target:
                poster_frame = out.copy()

            if progress_cb and (out_idx % max(1, fps) == 0 or out_idx + 1 == total_out):
                progress_cb(int((out_idx + 1) / total_out * 100), "rendering_game_video")

        writer.release()
        cap.release()
        if not os.path.exists(silent_path) or os.path.getsize(silent_path) < 1000:
            raise RuntimeError("Stylized video renderer produced no output")
        _mux_audio(silent_path, audio_path, output_path)

    if poster_path and poster_frame is not None:
        # Dark bottom gradient makes white TV menu text readable over every theme.
        poster = poster_frame.copy()
        ph, pw = poster.shape[:2]
        overlay = np.zeros_like(poster)
        alpha = np.linspace(0, .45, ph, dtype=np.float32)[:, None, None]
        poster = (poster.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha).astype(np.uint8)
        cv2.imwrite(poster_path, poster, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    primary_backend = max(backends, key=backends.get) if backends else "pose_fallback"
    return {
        "video_path": output_path,
        "poster_path": poster_path or "",
        "theme": theme.as_dict(),
        "segmentation_backend": primary_backend,
        "segmentation_usage": backends,
        "width": width,
        "height": height,
        "fps": fps,
    }
