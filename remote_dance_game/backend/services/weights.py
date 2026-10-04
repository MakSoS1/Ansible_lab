import numpy as np
from typing import List, Dict, Any
from config import WEIGHT_WINDOW_MS, BODY_PART_WEIGHTS_RANGE, ARMS_JOINTS, LEGS_JOINTS, TORSO_JOINTS


def generate_weights(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not frames:
        return []

    fps = 30
    window_frames = max(1, int(WEIGHT_WINDOW_MS / 1000.0 * fps))
    weights_list = []
    n = len(frames)

    for start in range(0, n, window_frames):
        end = min(start + window_frames, n)
        window = frames[start:end]

        arms_energy = np.mean([f.get("motion_energy", {}).get("arms", 0) for f in window])
        legs_energy = np.mean([f.get("motion_energy", {}).get("legs", 0) for f in window])
        torso_energy = np.mean([f.get("motion_energy", {}).get("torso", 0) for f in window])

        max_e = max(arms_energy, legs_energy, torso_energy, 0.001)

        def scale(energy, min_w, max_w):
            if max_e < 0.001:
                return (min_w + max_w) / 2
            norm = energy / max_e
            return round(min_w + norm * (max_w - min_w), 3)

        arms_range = BODY_PART_WEIGHTS_RANGE["arms"]
        legs_range = BODY_PART_WEIGHTS_RANGE["legs"]
        torso_range = BODY_PART_WEIGHTS_RANGE["torso"]

        w = {
            "start_ms": window[0]["t_ms"],
            "end_ms": window[-1]["t_ms"],
            "arms": scale(arms_energy, *arms_range),
            "legs": scale(legs_energy, *legs_range),
            "torso": scale(torso_energy, *torso_range),
        }
        weights_list.append(w)

    return weights_list


def get_weights_at_time(weights_list: List[Dict], t_ms: int) -> Dict[str, float]:
    default = {"arms": 0.5, "legs": 0.5, "torso": 0.3}
    for w in weights_list:
        if w["start_ms"] <= t_ms <= w["end_ms"]:
            return {"arms": w["arms"], "legs": w["legs"], "torso": w["torso"]}
    if weights_list:
        last = weights_list[-1]
        return {"arms": last["arms"], "legs": last["legs"], "torso": last["torso"]}
    return default
