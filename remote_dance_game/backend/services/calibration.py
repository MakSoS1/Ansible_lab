import numpy as np
from typing import List, Dict, Any
from models.schemas import CalibrationCheck, CalibrationResult
from config import CALIBRATION_THRESHOLDS


KEY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
FRAME_JOINTS = [11, 12, 15, 16, 23, 24, 27, 28]


def run_calibration(landmarks_history: List[List[Dict]]) -> CalibrationResult:
    if not landmarks_history or len(landmarks_history) < 5:
        return CalibrationResult(
            checks=[CalibrationCheck(name="data", passed=False,
                                     message="Not enough data. Stand still in front of the camera.")],
            ready=False,
        )

    last_frames = landmarks_history[-10:]

    all_vis = []
    all_xy = []
    for frame_lms in last_frames:
        if frame_lms and len(frame_lms) >= 33:
            vis = [l.get("v", 0) for l in frame_lms[:33]]
            all_vis.append(vis)
            all_xy.append([(l.get("x", 0.0), l.get("y", 0.0)) for l in frame_lms[:33]])

    if not all_vis:
        return CalibrationResult(checks=[], ready=False)

    avg_vis = np.mean(all_vis, axis=0)
    key_vis = [avg_vis[i] for i in KEY_JOINTS if i < len(avg_vis)]
    visible_key = sum(1 for v in key_vis if v >= CALIBRATION_THRESHOLDS["min_visibility"])

    checks = []

    visibility_passed = visible_key >= CALIBRATION_THRESHOLDS["min_key_joints_visible"]
    checks.append(CalibrationCheck(
        name="body_visibility",
        passed=visibility_passed,
        message="All key joints visible" if visibility_passed else
                f"Only {visible_key}/{len(KEY_JOINTS)} key joints visible. Make sure your full body is in frame.",
    ))

    valid_points = 0
    total_points = 0
    for frame_xy in all_xy:
        for ji in FRAME_JOINTS:
            if ji < len(frame_xy):
                x, y = frame_xy[ji]
                total_points += 1
                if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                    valid_points += 1
    in_frame_ratio = (valid_points / total_points) if total_points else 0.0
    in_frame = in_frame_ratio >= 0.75
    checks.append(CalibrationCheck(
        name="in_frame",
        passed=in_frame,
        message="Positioned well in frame" if in_frame else
                "You may be too close or too far from the camera. Step back a bit.",
    ))

    if len(all_vis) >= 5:
        jitter = np.mean([np.std([v[j] for v in all_vis[-5:]]) for j in KEY_JOINTS if j < len(all_vis[0])])
        stable = jitter < CALIBRATION_THRESHOLDS["max_jitter"]
    else:
        stable = True
    checks.append(CalibrationCheck(
        name="stability",
        passed=stable,
        message="Tracking is stable" if stable else
                "Tracking is unstable. Try better lighting and avoid moving the phone.",
    ))

    leg_vis = min(avg_vis[27] if 27 < len(avg_vis) else 0,
                  avg_vis[28] if 28 < len(avg_vis) else 0)
    legs_ok = leg_vis >= 0.5
    checks.append(CalibrationCheck(
        name="legs_visible",
        passed=legs_ok,
        message="Legs are visible" if legs_ok else
                "Legs not clearly visible. Lower the phone or step back so your full body is in frame.",
    ))

    avg_confidence = float(np.mean(avg_vis))
    light_ok = avg_confidence > 0.45
    checks.append(CalibrationCheck(
        name="lighting",
        passed=light_ok,
        message="Lighting is good" if light_ok else
                "Lighting may be too dark. Turn on more lights.",
    ))

    ready = all(c.passed for c in checks)
    return CalibrationResult(checks=checks, ready=ready)
