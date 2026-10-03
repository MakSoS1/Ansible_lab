import math
from typing import Any, Dict, List, Optional

import numpy as np

from config import ARMS_JOINTS, JOINT_ANGLE_DEFS, KEY_VELOCITY_JOINTS, LEGS_JOINTS, TORSO_JOINTS


def _point(landmarks: List[Dict[str, Any]], idx: int) -> np.ndarray:
    if idx >= len(landmarks):
        return np.zeros(3, dtype=np.float64)
    lm = landmarks[idx]
    return np.array([float(lm.get("x", 0.0)), float(lm.get("y", 0.0)), float(lm.get("z", 0.0))], dtype=np.float64)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return 0.0
    cos_v = float(np.dot(ba, bc) / denom)
    cos_v = max(-1.0, min(1.0, cos_v))
    return float(np.degrees(np.arccos(cos_v)))


def compute_angles(landmarks: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(landmarks) < 33:
        return {}
    return {
        name: round(_angle(_point(landmarks, a), _point(landmarks, b), _point(landmarks, c)), 2)
        for name, (a, b, c) in JOINT_ANGLE_DEFS.items()
    }


def compute_velocities(
    landmarks: List[Dict[str, Any]],
    previous_landmarks: Optional[List[Dict[str, Any]]],
    dt_seconds: float,
) -> Dict[str, List[float]]:
    velocities: Dict[str, List[float]] = {}
    if len(landmarks) < 33:
        return velocities

    if not previous_landmarks or len(previous_landmarks) < 33 or dt_seconds <= 1e-4 or dt_seconds > 0.5:
        return {f"joint_{idx}": [0.0, 0.0, 0.0] for idx in KEY_VELOCITY_JOINTS}

    for idx in KEY_VELOCITY_JOINTS:
        velocity = (_point(landmarks, idx) - _point(previous_landmarks, idx)) / dt_seconds
        velocities[f"joint_{idx}"] = [round(float(v), 4) for v in velocity]
    return velocities


def compute_motion_energy(velocities: Dict[str, List[float]]) -> Dict[str, float]:
    energy: Dict[str, float] = {"arms": 0.0, "legs": 0.0, "torso": 0.0}
    for part, joints in (("arms", ARMS_JOINTS), ("legs", LEGS_JOINTS), ("torso", TORSO_JOINTS)):
        values = []
        for idx in joints:
            value = velocities.get(f"joint_{idx}")
            if value is None:
                continue
            values.append(float(value[0]) ** 2 + float(value[1]) ** 2 + float(value[2]) ** 2)
        if values:
            energy[part] = round(float(sum(values) / len(values)), 4)
    return energy


def visibility_score(landmarks: List[Dict[str, Any]]) -> float:
    if not landmarks:
        return 0.0
    visible = [float(lm.get("v", lm.get("visibility", 0.0))) for lm in landmarks[:33]]
    if not visible:
        return 0.0
    key_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    selected = [visible[i] for i in key_indices if i < len(visible)]
    return round(float(sum(selected) / max(len(selected), 1)), 4)
