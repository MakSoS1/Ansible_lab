import cv2
import math
import os
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import CACHE_DIR, POSE_EXTRACTION_FPS

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_FILE = "pose_landmarker_heavy.task"
LEFT_RIGHT_SWAP_PAIRS = [
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]

_EMPTY_LM = {"x": 0.0, "y": 0.0, "z": 0.0, "v": 0.0}


def _get_model_path() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_path = os.path.join(CACHE_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"Downloading pose model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Model downloaded.")
    return model_path


def _blank_frame(t_ms: int) -> Dict[str, Any]:
    return {
        "t_ms": int(t_ms),
        "landmarks": [dict(_EMPTY_LM) for _ in range(33)],
        "world_landmarks": [dict(_EMPTY_LM) for _ in range(33)],
    }


def _to_frame(t_ms: int, pose_landmarks, world_landmarks, mirror_mode: bool) -> Dict[str, Any]:
    frame_entry = {"t_ms": int(t_ms), "landmarks": [], "world_landmarks": []}

    for lm in pose_landmarks[:33]:
        frame_entry["landmarks"].append({
            "x": round(float(lm.x), 4),
            "y": round(float(lm.y), 4),
            "z": round(float(lm.z), 4),
            "v": round(float(getattr(lm, "visibility", getattr(lm, "presence", 1.0))), 4),
        })
    while len(frame_entry["landmarks"]) < 33:
        frame_entry["landmarks"].append(dict(_EMPTY_LM))

    if world_landmarks:
        for lm in world_landmarks[:33]:
            frame_entry["world_landmarks"].append({
                "x": round(float(lm.x), 4),
                "y": round(float(lm.y), 4),
                "z": round(float(lm.z), 4),
                "v": round(float(getattr(lm, "visibility", getattr(lm, "presence", 1.0))), 4),
            })
    while len(frame_entry["world_landmarks"]) < 33:
        frame_entry["world_landmarks"].append(dict(_EMPTY_LM))

    return mirror_landmarks(frame_entry) if mirror_mode else frame_entry


def _pose_center_and_scale(frame: Dict[str, Any]) -> Tuple[float, float, float]:
    lm = frame.get("landmarks") or []
    if len(lm) < 29:
        return 0.5, 0.5, 0.1

    def point(idx: int) -> Tuple[float, float]:
        p = lm[idx]
        return float(p.get("x", 0.5)), float(p.get("y", 0.5))

    lx, ly = point(23)
    rx, ry = point(24)
    sx1, sy1 = point(11)
    sx2, sy2 = point(12)
    ax1, ay1 = point(27)
    ax2, ay2 = point(28)

    cx = (lx + rx) * 0.5
    cy = ((ly + ry) * 0.5 + (sy1 + sy2) * 0.5) * 0.5
    shoulder_mid = ((sx1 + sx2) * 0.5, (sy1 + sy2) * 0.5)
    hip_mid = ((lx + rx) * 0.5, (ly + ry) * 0.5)
    ankle_mid = ((ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5)
    torso = math.dist(shoulder_mid, hip_mid)
    height = math.dist(shoulder_mid, ankle_mid)
    scale = max(torso, height * 0.35, 0.05)
    return cx, cy, scale


def _mean_visibility(frame: Dict[str, Any]) -> float:
    lm = frame.get("landmarks") or []
    idxs = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    vals = [float(lm[i].get("v", 0.0)) for i in idxs if i < len(lm)]
    return sum(vals) / max(len(vals), 1)


def _assign_detections(
    detections: List[Dict[str, Any]],
    active: Dict[int, Dict[str, Any]],
    t_ms: int,
    max_tracks: int,
    next_track_id: int,
) -> Tuple[Dict[int, Dict[str, Any]], int]:
    """Greedy identity association for spatially separated dancers.

    PoseLandmarker returns multiple people but no persistent person ID.  A
    centre/scale tracker is enough for a dance stage where performers normally
    keep stable left/right lanes.  It also avoids identity switches caused by a
    change in MediaPipe detection ordering.
    """
    det_meta = []
    for det in detections:
        cx, cy, scale = _pose_center_and_scale(det)
        det_meta.append((det, cx, cy, scale))

    candidates: List[Tuple[float, int, int]] = []
    for track_id, state in active.items():
        if t_ms - int(state.get("last_seen_ms", -100000)) > 850:
            continue
        for det_idx, (_, cx, cy, scale) in enumerate(det_meta):
            prev_cx = float(state.get("cx", cx))
            prev_cy = float(state.get("cy", cy))
            prev_scale = max(float(state.get("scale", scale)), 0.05)
            spatial = math.hypot(cx - prev_cx, cy - prev_cy)
            scale_delta = abs(scale - prev_scale) / max(prev_scale, scale, 0.05)
            cost = spatial + 0.12 * scale_delta
            candidates.append((cost, track_id, det_idx))

    assignments: Dict[int, Dict[str, Any]] = {}
    used_tracks = set()
    used_dets = set()
    for cost, track_id, det_idx in sorted(candidates, key=lambda x: x[0]):
        if cost > 0.28 or track_id in used_tracks or det_idx in used_dets:
            continue
        assignments[track_id] = det_meta[det_idx][0]
        used_tracks.add(track_id)
        used_dets.add(det_idx)

    for det_idx, (det, cx, cy, scale) in enumerate(det_meta):
        if det_idx in used_dets:
            continue
        if len(active) >= max_tracks:
            break
        track_id = next_track_id
        next_track_id += 1
        active[track_id] = {
            "track_id": track_id,
            "cx": cx,
            "cy": cy,
            "scale": scale,
            "last_seen_ms": t_ms,
        }
        assignments[track_id] = det
        used_tracks.add(track_id)

    for track_id, det in assignments.items():
        cx, cy, scale = _pose_center_and_scale(det)
        state = active.setdefault(track_id, {"track_id": track_id})
        # Smooth the tracking coordinates only; actual landmarks remain raw.
        state["cx"] = 0.68 * float(state.get("cx", cx)) + 0.32 * cx
        state["cy"] = 0.68 * float(state.get("cy", cy)) + 0.32 * cy
        state["scale"] = 0.72 * float(state.get("scale", scale)) + 0.28 * scale
        state["last_seen_ms"] = t_ms

    return assignments, next_track_id


def extract_poses_from_video(
    video_path: str,
    fps: int = POSE_EXTRACTION_FPS,
    progress_cb: Optional[Callable] = None,
    mirror_mode: bool = True,
    max_poses: int = 4,
) -> Dict[str, Any]:
    import mediapipe as mp

    model_path = _get_model_path()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_ms = int(total_frames / video_fps * 1000)

    if progress_cb:
        progress_cb(0, "pose_extraction")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=max(1, min(int(max_poses), 6)),
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    # Track frames are backfilled with blanks when a dancer enters later.
    track_frames: Dict[int, List[Dict[str, Any]]] = {}
    track_stats: Dict[int, Dict[str, Any]] = {}
    active: Dict[int, Dict[str, Any]] = {}
    next_track_id = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t_ms = int(frame_idx / video_fps * 1000)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, t_ms)

        detections: List[Dict[str, Any]] = []
        count = len(result.pose_landmarks or [])
        for pose_idx in range(count):
            world = None
            if result.pose_world_landmarks and pose_idx < len(result.pose_world_landmarks):
                world = result.pose_world_landmarks[pose_idx]
            detections.append(_to_frame(
                t_ms,
                result.pose_landmarks[pose_idx],
                world,
                mirror_mode,
            ))

        assignments, next_track_id = _assign_detections(
            detections, active, t_ms, max(1, min(int(max_poses), 6)), next_track_id
        )

        # Append one frame to every known track so all timelines stay aligned.
        for track_id in list(active.keys()):
            frames = track_frames.setdefault(track_id, [_blank_frame(int(i / video_fps * 1000)) for i in range(frame_idx)])
            assigned = assignments.get(track_id)
            frames.append(assigned if assigned is not None else _blank_frame(t_ms))

            stat = track_stats.setdefault(track_id, {
                "visible_frames": 0,
                "visibility_sum": 0.0,
                "x_sum": 0.0,
                "x_count": 0,
                "scale_sum": 0.0,
            })
            if assigned is not None:
                vis = _mean_visibility(assigned)
                cx, _, scale = _pose_center_and_scale(assigned)
                if vis >= 0.25:
                    stat["visible_frames"] += 1
                stat["visibility_sum"] += vis
                stat["x_sum"] += cx
                stat["x_count"] += 1
                stat["scale_sum"] += scale

        frame_idx += 1
        if progress_cb and frame_idx % 30 == 0:
            pct = int(frame_idx / max(total_frames, 1) * 100)
            progress_cb(min(pct, 99), "pose_extraction")

    cap.release()
    landmarker.close()

    for track_id, frames in track_frames.items():
        while len(frames) < frame_idx:
            t_ms = int(len(frames) / video_fps * 1000)
            frames.append(_blank_frame(t_ms))

    candidates = []
    for track_id, frames in track_frames.items():
        stat = track_stats.get(track_id, {})
        coverage = float(stat.get("visible_frames", 0)) / max(frame_idx, 1)
        avg_vis = float(stat.get("visibility_sum", 0.0)) / max(int(stat.get("x_count", 0)), 1)
        avg_x = float(stat.get("x_sum", 0.0)) / max(int(stat.get("x_count", 0)), 1)
        avg_scale = float(stat.get("scale_sum", 0.0)) / max(int(stat.get("x_count", 0)), 1)
        quality = coverage * (0.55 + min(avg_vis, 1.0) * 0.45) * (0.55 + min(avg_scale * 3.2, 1.0) * 0.45)
        if coverage >= 0.10:
            candidates.append({
                "track_id": track_id,
                "frames": frames,
                "coverage": round(coverage, 4),
                "avg_visibility": round(avg_vis, 4),
                "avg_x": round(avg_x, 4),
                "avg_scale": round(avg_scale, 4),
                "quality": round(quality, 5),
            })

    # Do not turn a brief background person into a selectable coach.  Keep
    # tracks that are substantial relative to the strongest stage performer.
    viable = []
    if candidates:
        best_quality = max(float(tr["quality"]) for tr in candidates)
        best_scale = max(float(tr["avg_scale"]) for tr in candidates)
        for tr in candidates:
            if float(tr["coverage"]) < 0.20:
                continue
            if float(tr["avg_visibility"]) < 0.28:
                continue
            if float(tr["avg_scale"]) < max(0.045, best_scale * 0.45):
                continue
            if float(tr["quality"]) < best_quality * 0.32:
                continue
            viable.append(tr)

    viable = sorted(viable, key=lambda tr: float(tr["quality"]), reverse=True)[:max_poses]
    # Stable role ordering: left-to-right in the mirrored scoring space.
    viable.sort(key=lambda tr: (tr["avg_x"], -tr["coverage"]))
    for role_index, tr in enumerate(viable):
        tr["role_index"] = role_index

    if viable:
        # Default/legacy reference uses the most reliable dancer, not arbitrary
        # MediaPipe detection index. Multi-coach gameplay can explicitly choose.
        primary = max(
            viable,
            key=lambda tr: tr["coverage"] * (0.65 + 0.35 * tr["avg_visibility"]) * (0.8 + min(tr["avg_scale"], 0.7)),
        )
        primary_frames = primary["frames"]
        primary_role_index = int(primary["role_index"])
    else:
        primary_frames = [_blank_frame(int(i / video_fps * 1000)) for i in range(frame_idx)]
        primary_role_index = 0

    if progress_cb:
        progress_cb(100, "pose_extraction")

    return {
        "fps": int(video_fps),
        "total_frames": frame_idx,
        "duration_ms": duration_ms,
        "frames": primary_frames,
        "tracks": viable,
        "primary_role_index": primary_role_index,
        "coach_count": max(1, len(viable)),
    }


def mirror_landmarks(frame_entry: dict) -> dict:
    lm = frame_entry.get("landmarks", [])
    if len(lm) < 33:
        return frame_entry

    mirrored_lm = [dict(l) for l in lm[:33]]
    for l in mirrored_lm:
        l["x"] = round(1.0 - float(l["x"]), 4)
    for i, j in LEFT_RIGHT_SWAP_PAIRS:
        mirrored_lm[i], mirrored_lm[j] = mirrored_lm[j], mirrored_lm[i]

    world_lm = frame_entry.get("world_landmarks", [])
    mirrored_world = []
    if len(world_lm) >= 33:
        mirrored_world = [dict(l) for l in world_lm[:33]]
        for l in mirrored_world:
            l["x"] = round(-float(l["x"]), 4)
        for i, j in LEFT_RIGHT_SWAP_PAIRS:
            mirrored_world[i], mirrored_world[j] = mirrored_world[j], mirrored_world[i]

    return {
        "t_ms": frame_entry["t_ms"],
        "landmarks": mirrored_lm,
        "world_landmarks": mirrored_world,
    }
