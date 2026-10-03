import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from typing import List, Dict, Any, Optional
import math
from config import (
    POSE_SMOOTHING_WINDOW, POSE_SMOOTHING_ORDER,
    JOINT_ANGLE_DEFS, LIMB_VECTOR_DEFS, KEY_VELOCITY_JOINTS,
    ARMS_JOINTS, LEGS_JOINTS, TORSO_JOINTS,
)


def process_choreography(raw_pose_data: Dict[str, Any]) -> Dict[str, Any]:
    frames = raw_pose_data["frames"]
    if not frames:
        return raw_pose_data

    frames = interpolate_missing(frames)
    frames = smooth_poses(frames)
    frames = compute_angles(frames)
    frames = compute_velocities(frames)
    frames = compute_motion_energy(frames)

    raw_pose_data["frames"] = frames
    return raw_pose_data


def interpolate_missing(frames: List[Dict]) -> List[Dict]:
    missing_indices = []
    for i, f in enumerate(frames):
        if not f["landmarks"] or f["landmarks"][0].get("v", 0) < 0.01:
            missing_indices.append(i)

    if not missing_indices or len(missing_indices) >= len(frames) - 2:
        return frames

    good_indices = [i for i in range(len(frames)) if i not in missing_indices]
    if len(good_indices) < 2:
        return frames

    for lm_key in ["landmarks", "world_landmarks"]:
        for joint_idx in range(33):
            good_t = [frames[i]["t_ms"] for i in good_indices]
            good_vals = []
            for i in good_indices:
                lms = frames[i].get(lm_key, [])
                if joint_idx < len(lms):
                    good_vals.append(lms[joint_idx])
                else:
                    good_vals.append({"x": 0, "y": 0, "z": 0, "v": 0})

            if len(good_vals) < 4:
                continue

            for coord in ["x", "y", "z"]:
                vals = [g[coord] for g in good_vals]
                try:
                    interp_fn = interp1d(good_t, vals, kind="linear",
                                         fill_value="extrapolate")
                    for mi in missing_indices:
                        t = frames[mi]["t_ms"]
                        lms = frames[mi].get(lm_key, [])
                        if joint_idx < len(lms):
                            lms[joint_idx][coord] = round(float(interp_fn(t)), 4)
                except Exception:
                    pass

            vis_interp = min(good_vals, key=lambda g: g["v"])["v"] * 0.5
            for mi in missing_indices:
                lms = frames[mi].get(lm_key, [])
                if joint_idx < len(lms):
                    lms[joint_idx]["v"] = round(vis_interp, 4)

    return frames


def smooth_poses(frames: List[Dict]) -> List[Dict]:
    n = len(frames)
    if n < POSE_SMOOTHING_WINDOW:
        return frames

    half_w = POSE_SMOOTHING_WINDOW // 2

    for lm_key in ["landmarks", "world_landmarks"]:
        for joint_idx in range(33):
            for coord in ["x", "y", "z"]:
                vals = []
                for f in frames:
                    lms = f.get(lm_key, [])
                    if joint_idx < len(lms):
                        vals.append(lms[joint_idx][coord])
                    else:
                        vals.append(0)

                try:
                    smoothed = savgol_filter(vals, POSE_SMOOTHING_WINDOW,
                                             POSE_SMOOTHING_ORDER)
                    for i in range(half_w, n - half_w):
                        lms = frames[i].get(lm_key, [])
                        if joint_idx < len(lms):
                            lms[joint_idx][coord] = round(float(smoothed[i]), 4)
                except Exception:
                    pass

    return frames


def _landmark_to_np(landmarks: List[Dict], idx: int) -> np.ndarray:
    if idx < len(landmarks):
        return np.array([landmarks[idx]["x"], landmarks[idx]["y"], landmarks[idx]["z"]])
    return np.array([0.0, 0.0, 0.0])


def _compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return round(float(np.degrees(np.arccos(cos_val))), 1)


def compute_angles(frames: List[Dict]) -> List[Dict]:
    for f in frames:
        lm = f.get("landmarks", [])
        angles = {}
        for name, (a_idx, b_idx, c_idx) in JOINT_ANGLE_DEFS.items():
            a = _landmark_to_np(lm, a_idx)
            b = _landmark_to_np(lm, b_idx)
            c = _landmark_to_np(lm, c_idx)
            angles[name] = _compute_angle(a, b, c)
        f["angles"] = angles
    return frames


def compute_velocities(frames: List[Dict]) -> List[Dict]:
    n = len(frames)
    for i in range(n):
        f = frames[i]
        velocities = {}

        if i > 0:
            prev = frames[i - 1]
            dt = (f["t_ms"] - prev["t_ms"]) / 1000.0
            if dt > 0:
                lm = f.get("landmarks", [])
                prev_lm = prev.get("landmarks", [])
                for jidx in KEY_VELOCITY_JOINTS:
                    cur = _landmark_to_np(lm, jidx)
                    prv = _landmark_to_np(prev_lm, jidx)
                    vel = (cur - prv) / dt
                    velocities[f"joint_{jidx}"] = [round(float(v), 4) for v in vel]
        else:
            for jidx in KEY_VELOCITY_JOINTS:
                velocities[f"joint_{jidx}"] = [0.0, 0.0, 0.0]

        f["velocities"] = velocities
    return frames


def compute_motion_energy(frames: List[Dict]) -> List[Dict]:
    for i in range(len(frames)):
        f = frames[i]
        energy = {"arms": 0.0, "legs": 0.0, "torso": 0.0}

        if i > 0 and f.get("velocities"):
            prev = frames[i - 1]
            lm = f.get("landmarks", [])

            for part_name, joint_list in [("arms", ARMS_JOINTS), ("legs", LEGS_JOINTS), ("torso", TORSO_JOINTS)]:
                total_v = 0.0
                count = 0
                for jidx in joint_list:
                    key = f"joint_{jidx}"
                    if key in f.get("velocities", {}):
                        v = f["velocities"][key]
                        total_v += v[0]**2 + v[1]**2 + v[2]**2
                        count += 1
                if count > 0:
                    energy[part_name] = round(total_v / count, 4)

        f["motion_energy"] = energy
    return frames
