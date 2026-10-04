import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from config import (
    JOINT_ANGLE_DEFS, LIMB_VECTOR_DEFS, SCORING_WEIGHTS,
    ARMS_JOINTS, LEGS_JOINTS, TORSO_JOINTS,
)


def _lm_to_np(landmarks: List[Dict], idx: int) -> np.ndarray:
    if idx < len(landmarks):
        return np.array([landmarks[idx]["x"], landmarks[idx]["y"], landmarks[idx]["z"]])
    return np.zeros(3)


def limb_vector_similarity(ref_lm: List[Dict], player_lm: List[Dict],
                           ref_vis: List[float], player_vis: List[float]) -> float:
    scores = []
    weights = []

    for name, (a_idx, b_idx) in LIMB_VECTOR_DEFS.items():
        ref_a = _lm_to_np(ref_lm, a_idx)
        ref_b = _lm_to_np(ref_lm, b_idx)
        play_a = _lm_to_np(player_lm, a_idx)
        play_b = _lm_to_np(player_lm, b_idx)

        ref_vec = ref_b - ref_a
        play_vec = play_b - play_a

        ref_norm = np.linalg.norm(ref_vec)
        play_norm = np.linalg.norm(play_vec)

        if ref_norm < 1e-6 or play_norm < 1e-6:
            continue

        ref_dir = ref_vec / ref_norm
        play_dir = play_vec / play_norm

        cos_sim = float(np.dot(ref_dir, play_dir))
        cos_sim = max(-1.0, min(1.0, cos_sim))
        sim = (cos_sim + 1.0) / 2.0

        len_diff = abs(ref_norm - play_norm) / max(ref_norm, play_norm, 1e-6)
        len_sim = 1.0 - min(len_diff, 1.0)

        combined = 0.7 * sim + 0.3 * len_sim

        min_vis = min(ref_vis[a_idx] if a_idx < len(ref_vis) else 0,
                      ref_vis[b_idx] if b_idx < len(ref_vis) else 0,
                      player_vis[a_idx] if a_idx < len(player_vis) else 0,
                      player_vis[b_idx] if b_idx < len(player_vis) else 0)

        conf_weight = max(0.1, min_vis)
        scores.append(combined * conf_weight)
        weights.append(conf_weight)

    if not weights or sum(weights) < 0.01:
        return 0.5

    return sum(scores) / sum(weights)


def joint_angle_similarity(ref_angles: Dict[str, float],
                           player_angles: Dict[str, float],
                           ref_vis: List[float],
                           player_vis: List[float]) -> float:
    scores = []
    weights = []

    for name, (a_idx, b_idx, c_idx) in JOINT_ANGLE_DEFS.items():
        if name in ref_angles and name in player_angles:
            ref_a = ref_angles[name]
            play_a = player_angles[name]
            diff = abs(ref_a - play_a)
            norm_diff = diff / 180.0
            sim = 1.0 - min(norm_diff, 1.0)

            min_vis = min(ref_vis[a_idx] if a_idx < len(ref_vis) else 0,
                          ref_vis[b_idx] if b_idx < len(ref_vis) else 0,
                          ref_vis[c_idx] if c_idx < len(ref_vis) else 0,
                          player_vis[a_idx] if a_idx < len(player_vis) else 0,
                          player_vis[b_idx] if b_idx < len(player_vis) else 0,
                          player_vis[c_idx] if c_idx < len(player_vis) else 0)

            conf_weight = max(0.1, min_vis)
            scores.append(sim * conf_weight)
            weights.append(conf_weight)

    if not weights or sum(weights) < 0.01:
        return 0.5

    return sum(scores) / sum(weights)


def motion_similarity(ref_vel: Dict[str, List[float]],
                      player_vel: Dict[str, List[float]],
                      ref_vis: List[float],
                      player_vis: List[float]) -> float:
    from config import KEY_VELOCITY_JOINTS

    scores = []
    weights = []

    for jidx in KEY_VELOCITY_JOINTS:
        key = f"joint_{jidx}"
        if key in ref_vel and key in player_vel:
            rv = np.array(ref_vel[key])
            pv = np.array(player_vel[key])

            r_norm = np.linalg.norm(rv)
            p_norm = np.linalg.norm(pv)

            if r_norm < 1e-6 and p_norm < 1e-6:
                weight = 0.5
                scores.append(1.0 * weight)
                weights.append(weight)
                continue

            if r_norm < 1e-6 or p_norm < 1e-6:
                weight = 0.3
                scores.append(0.2 * weight)
                weights.append(weight)
                continue

            cos_sim = float(np.dot(rv, pv) / (r_norm * p_norm))
            cos_sim = max(-1.0, min(1.0, cos_sim))
            dir_sim = (cos_sim + 1.0) / 2.0

            norm_diff = abs(r_norm - p_norm) / max(r_norm, p_norm, 1e-6)
            mag_sim = 1.0 - min(norm_diff, 1.0)

            combined = 0.6 * dir_sim + 0.4 * mag_sim

            min_vis = min(ref_vis[jidx] if jidx < len(ref_vis) else 0,
                          player_vis[jidx] if jidx < len(player_vis) else 0)
            conf_weight = max(0.1, min_vis)
            scores.append(combined * conf_weight)
            weights.append(conf_weight)

    if not weights or sum(weights) < 0.01:
        return 0.5

    return sum(scores) / sum(weights)


def timing_similarity(time_offset_ms: float, window_ms: float = 120.0) -> float:
    sigma = window_ms / 2.0
    return float(np.exp(-(time_offset_ms ** 2) / (2 * sigma ** 2)))


def compute_part_scores(ref_lm: List[Dict], player_lm: List[Dict],
                        ref_vis: List[float], player_vis: List[float]) -> Dict[str, float]:
    part_scores = {}

    for part_name, joint_list in [("arms", ARMS_JOINTS), ("legs", LEGS_JOINTS), ("torso", TORSO_JOINTS)]:
        dists = []
        for jidx in joint_list:
            if jidx < len(ref_lm) and jidx < len(player_lm):
                r = _lm_to_np(ref_lm, jidx)
                p = _lm_to_np(player_lm, jidx)
                dist = np.linalg.norm(r - p)
                sim = max(0.0, 1.0 - dist * 2.0)
                min_v = min(ref_vis[jidx] if jidx < len(ref_vis) else 0,
                            player_vis[jidx] if jidx < len(player_vis) else 0)
                if min_v > 0.3:
                    dists.append(sim)
        if dists:
            part_scores[part_name] = round(float(np.mean(dists)), 3)
        else:
            part_scores[part_name] = 0.0

    return part_scores


def compare_poses(ref_norm: Dict[str, Any], player_norm: Dict[str, Any],
                  time_offset_ms: float = 0.0,
                  body_weights: Optional[Dict[str, float]] = None,
                  include_timing: bool = True) -> Tuple[float, Dict[str, float]]:
    if not ref_norm.get("valid") or not player_norm.get("valid"):
        return 0.0, {"arms": 0.0, "legs": 0.0, "torso": 0.0}

    ref_lm = ref_norm["landmarks"]
    play_lm = player_norm["landmarks"]
    ref_vis = [l.get("v", 0) for l in ref_lm[:33]]
    play_vis = [l.get("v", 0) for l in play_lm[:33]]

    limb_sim = limb_vector_similarity(ref_lm, play_lm, ref_vis, play_vis)
    angle_sim = joint_angle_similarity(
        ref_norm.get("angles", {}), player_norm.get("angles", {}),
        ref_vis, play_vis
    )
    motion_sim = motion_similarity(
        ref_norm.get("velocities", {}), player_norm.get("velocities", {}),
        ref_vis, play_vis
    )
    time_sim = timing_similarity(time_offset_ms)

    sw = SCORING_WEIGHTS
    pose_weight = sw["limb_vector"] + sw["joint_angle"] + sw["motion"]
    pose_score = (
        sw["limb_vector"] * limb_sim +
        sw["joint_angle"] * angle_sim +
        sw["motion"] * motion_sim
    ) / max(pose_weight, 1e-8)

    if body_weights:
        part_scores = compute_part_scores(ref_lm, play_lm, ref_vis, play_vis)
        weighted_part = sum(part_scores.get(k, 0.5) * body_weights.get(k, 0.5)
                            for k in ["arms", "legs", "torso"])
        total_w = sum(body_weights.get(k, 0.5) for k in ["arms", "legs", "torso"])
        if total_w > 0:
            weighted_part /= total_w
        pose_score = 0.7 * pose_score + 0.3 * weighted_part
    else:
        part_scores = compute_part_scores(ref_lm, play_lm, ref_vis, play_vis)

    if include_timing:
        total_weight = pose_weight + sw["timing"]
        timing_ratio = sw["timing"] / max(total_weight, 1e-8)
        raw_score = (1.0 - timing_ratio) * pose_score + timing_ratio * time_sim
    else:
        raw_score = pose_score

    return round(max(0.0, min(1.0, raw_score)), 4), part_scores
