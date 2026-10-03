import numpy as np
from typing import List, Dict, Any, Optional


HIP_INDICES = [23, 24]
SHOULDER_INDICES = [11, 12]


def normalize_pose(landmarks: List[Dict], world_landmarks: Optional[List[Dict]] = None) -> Dict[str, Any]:
    if not landmarks or len(landmarks) < 33:
        return {"landmarks": landmarks, "world_landmarks": world_landmarks or [], "valid": False}

    lm = np.array([[l.get("x", 0), l.get("y", 0), l.get("z", 0)] for l in landmarks[:33]])
    vis = np.array([l.get("v", 0) for l in landmarks[:33]])

    if vis[23] < 0.3 or vis[24] < 0.3:
        return {"landmarks": landmarks, "world_landmarks": world_landmarks or [], "valid": False}

    hip_mid = (lm[23] + lm[24]) / 2.0
    lm_centered = lm - hip_mid

    shoulder_mid = (lm[11] + lm[12]) / 2.0
    torso_vec = shoulder_mid - hip_mid
    torso_len = np.linalg.norm(torso_vec)

    if torso_len < 0.001:
        return {"landmarks": landmarks, "world_landmarks": world_landmarks or [], "valid": False}

    lm_scaled = lm_centered / torso_len

    shoulder_dir = lm[12] - lm[11]
    shoulder_angle = np.arctan2(shoulder_dir[1], shoulder_dir[0])
    cos_a = np.cos(-shoulder_angle)
    sin_a = np.sin(-shoulder_angle)
    rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    lm_final = (rot @ lm_scaled.T).T

    norm_lm = []
    for i in range(33):
        norm_lm.append({
            "x": round(float(lm_final[i, 0]), 4),
            "y": round(float(lm_final[i, 1]), 4),
            "z": round(float(lm_final[i, 2]), 4),
            "v": round(float(vis[i]), 4),
        })

    norm_wlm = []
    if world_landmarks and len(world_landmarks) >= 33:
        wlm = np.array([[l.get("x", 0), l.get("y", 0), l.get("z", 0)] for l in world_landmarks[:33]])
        wlm_centered = wlm - (wlm[23] + wlm[24]) / 2.0
        wlm_scaled = wlm_centered / torso_len
        wlm_final = (rot @ wlm_scaled.T).T

        for i in range(33):
            norm_wlm.append({
                "x": round(float(wlm_final[i, 0]), 4),
                "y": round(float(wlm_final[i, 1]), 4),
                "z": round(float(wlm_final[i, 2]), 4),
                "v": round(float(vis[i]), 4),
            })

    return {"landmarks": norm_lm, "world_landmarks": norm_wlm, "valid": True, "torso_len": float(torso_len)}


def normalize_frame_for_scoring(frame: Dict[str, Any]) -> Dict[str, Any]:
    lm = frame.get("landmarks", [])
    wlm = frame.get("world_landmarks", [])
    if len(lm) < 33:
        return {"valid": False}

    result = normalize_pose(lm, wlm if len(wlm) >= 33 else None)
    result["t_ms"] = frame.get("t_ms", 0)
    result["angles"] = frame.get("angles", {})
    result["velocities"] = frame.get("velocities", {})
    result["motion_energy"] = frame.get("motion_energy", {})
    return result
