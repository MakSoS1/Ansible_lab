import math
from typing import Any, Dict, List, Sequence

KEY_JOINTS = (13, 14, 15, 16, 25, 26, 27, 28)


def _distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    dx = float(a.get("x", 0.0)) - float(b.get("x", 0.0))
    dy = float(a.get("y", 0.0)) - float(b.get("y", 0.0))
    dz = float(a.get("z", 0.0)) - float(b.get("z", 0.0))
    return math.sqrt(dx * dx + dy * dy + 0.35 * dz * dz)


def _frame_motion(prev: Dict[str, Any], cur: Dict[str, Any]) -> float:
    a = prev.get("landmarks") or []
    b = cur.get("landmarks") or []
    vals = []
    for idx in KEY_JOINTS:
        if idx >= len(a) or idx >= len(b):
            continue
        if min(float(a[idx].get("v", 0.0)), float(b[idx].get("v", 0.0))) < 0.30:
            continue
        vals.append(_distance(a[idx], b[idx]))
    return sum(vals) / max(len(vals), 1)


def _nearest_frame_index(frames: Sequence[Dict[str, Any]], t_ms: int) -> int:
    if not frames:
        return 0
    lo, hi = 0, len(frames) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if int(frames[mid].get("t_ms", 0)) < t_ms:
            lo = mid + 1
        else:
            hi = mid
    idx = lo
    if idx > 0 and abs(int(frames[idx - 1].get("t_ms", 0)) - t_ms) < abs(int(frames[idx].get("t_ms", 0)) - t_ms):
        idx -= 1
    return idx


def build_move_markers(
    frames: List[Dict[str, Any]],
    timing: Dict[str, Any],
    min_gap_ms: int = 320,
    max_gap_ms: int = 650,
) -> List[Dict[str, Any]]:
    """Find readable, frequent choreography beats for grading and Next Moves.

    Markers are based on reference-body motion peaks, nudged toward musical beats,
    with a max-gap fill so quiet choreography still receives regular feedback.
    """
    if not frames:
        return []

    motion = [0.0]
    for i in range(1, len(frames)):
        motion.append(_frame_motion(frames[i - 1], frames[i]))

    sorted_motion = sorted(motion)
    threshold = sorted_motion[int((len(sorted_motion) - 1) * 0.52)] if sorted_motion else 0.0
    candidates: List[Dict[str, Any]] = []

    for i in range(2, max(2, len(frames) - 2)):
        value = motion[i]
        if value < max(0.0025, threshold):
            continue
        local = motion[max(0, i - 2): min(len(motion), i + 3)]
        if value < max(local):
            continue
        candidates.append({
            "t_ms": int(frames[i].get("t_ms", i * 33)),
            "frame_index": i,
            "motion": round(float(value), 5),
            "kind": "motion_peak",
        })

    # Musical beats help static / gesture-heavy choreography feel like a dance game.
    beats = timing.get("beat_ms") or []
    for beat in beats:
        t_ms = int(beat)
        idx = _nearest_frame_index(frames, t_ms)
        if idx >= len(frames):
            continue
        candidates.append({
            "t_ms": int(frames[idx].get("t_ms", t_ms)),
            "frame_index": idx,
            "motion": round(float(motion[idx]), 5),
            "kind": "beat",
        })

    candidates.sort(key=lambda x: (x["t_ms"], -x["motion"]))
    selected: List[Dict[str, Any]] = []
    for cand in candidates:
        if cand["t_ms"] < 250:
            continue
        if not selected or cand["t_ms"] - selected[-1]["t_ms"] >= min_gap_ms:
            selected.append(cand)
        elif cand["motion"] > selected[-1]["motion"] * 1.35:
            selected[-1] = cand

    end_ms = int(frames[-1].get("t_ms", 0))
    if not selected:
        selected.append({
            "t_ms": min(450, end_ms),
            "frame_index": _nearest_frame_index(frames, min(450, end_ms)),
            "motion": 0.0,
            "kind": "fallback",
        })

    # Never leave the player without feedback for too long.
    filled: List[Dict[str, Any]] = []
    previous_t = 150
    for marker in selected:
        while marker["t_ms"] - previous_t > max_gap_ms:
            fill_t = previous_t + min(500, max_gap_ms)
            idx = _nearest_frame_index(frames, fill_t)
            filled.append({
                "t_ms": int(frames[idx].get("t_ms", fill_t)),
                "frame_index": idx,
                "motion": round(float(motion[idx]), 5),
                "kind": "fill",
            })
            previous_t = filled[-1]["t_ms"]
        if not filled or marker["t_ms"] - filled[-1]["t_ms"] >= min_gap_ms:
            filled.append(marker)
        previous_t = filled[-1]["t_ms"]

    # Keep grading through the last meaningful seconds.
    while end_ms - previous_t > max_gap_ms and previous_t + min_gap_ms < end_ms:
        fill_t = min(previous_t + 500, end_ms - 120)
        idx = _nearest_frame_index(frames, fill_t)
        filled.append({
            "t_ms": int(frames[idx].get("t_ms", fill_t)),
            "frame_index": idx,
            "motion": round(float(motion[idx]), 5),
            "kind": "fill",
        })
        previous_t = filled[-1]["t_ms"]

    for i, marker in enumerate(filled):
        marker["move_index"] = i
    return filled


def _pose_novelty(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    la = a.get("landmarks") or []
    lb = b.get("landmarks") or []
    joints = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    vals = []
    for idx in joints:
        if idx >= len(la) or idx >= len(lb):
            continue
        if min(float(la[idx].get("v", 0.0)), float(lb[idx].get("v", 0.0))) < 0.25:
            continue
        vals.append(_distance(la[idx], lb[idx]))
    return sum(vals) / max(len(vals), 1)


def _pose_extent(frame: Dict[str, Any]) -> float:
    lm = frame.get("landmarks") or []
    if len(lm) < 29:
        return 0.0
    points = []
    for idx in (15, 16, 27, 28):
        if idx < len(lm) and float(lm[idx].get("v", 0.0)) >= 0.25:
            points.append((float(lm[idx].get("x", 0.0)), float(lm[idx].get("y", 0.0))))
    if len(points) < 2:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def _destination_keyframe(
    frames: List[Dict[str, Any]],
    motion: List[float],
    seed_index: int,
    previous_keyframe: Dict[str, Any] | None,
) -> int:
    """Choose the readable destination pose after a motion burst.

    Just Dance pictograms usually communicate where the body is going, not the
    instant of maximum velocity.  Search roughly 0.15-0.55 s after a motion
    peak and favour poses that are distinctive, extended and already settling.
    """
    if not frames:
        return 0

    seed_t = int(frames[seed_index].get("t_ms", 0))
    candidate_indices = []
    for i in range(seed_index, len(frames)):
        dt = int(frames[i].get("t_ms", 0)) - seed_t
        if dt < 140:
            continue
        if dt > 560:
            break
        candidate_indices.append(i)

    if not candidate_indices:
        return seed_index

    local_motion = [motion[i] for i in candidate_indices]
    max_motion = max(local_motion) if local_motion else 1.0
    max_extent = max((_pose_extent(frames[i]) for i in candidate_indices), default=1.0)

    best_idx = candidate_indices[0]
    best_score = -1e9
    anchor = previous_keyframe if previous_keyframe is not None else frames[max(0, seed_index - 8)]
    for i in candidate_indices:
        frame = frames[i]
        novelty = _pose_novelty(anchor, frame)
        settle = 1.0 - min(1.0, motion[i] / max(max_motion, 1e-6))
        extent = _pose_extent(frame) / max(max_extent, 1e-6)
        dt = int(frame.get("t_ms", 0)) - seed_t
        # Slight preference for ~300 ms after the velocity peak.
        temporal = max(0.0, 1.0 - abs(dt - 310) / 360.0)
        score = novelty * 1.55 + settle * 0.52 + extent * 0.36 + temporal * 0.18
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def build_pictogram_markers(
    frames: List[Dict[str, Any]],
    timing: Dict[str, Any],
    min_gap_ms: int = 1200,
    max_gap_ms: int = 2400,
) -> List[Dict[str, Any]]:
    """Sparse Just-Dance-style visual instructions.

    Scoring can remain dense, but pictograms must be slower and legible.  Each
    cue is based on a real reference dancer and depicts a destination key pose
    after a motion burst rather than the raw maximum-velocity frame.
    """
    if not frames:
        return []

    motion = [0.0]
    for i in range(1, len(frames)):
        motion.append(_frame_motion(frames[i - 1], frames[i]))

    # Pictograms are instructions, not a pose monitor. Prefer only the most
    # meaningful movement bursts so the player has time to read each cue.
    sorted_motion = sorted(motion)
    threshold = sorted_motion[int((len(sorted_motion) - 1) * 0.68)] if sorted_motion else 0.0

    seeds: List[int] = []
    for i in range(2, max(2, len(frames) - 2)):
        value = motion[i]
        if value < max(0.0028, threshold):
            continue
        local = motion[max(0, i - 2): min(len(motion), i + 3)]
        if value >= max(local):
            seeds.append(i)

    # If pose motion is unusually subtle, strong musical beats are only a
    # fallback. They must not turn the pictogram rail into a beat visualizer.
    if len(seeds) < 2:
        for beat in timing.get("strong_beat_ms") or timing.get("beat_ms") or []:
            seeds.append(_nearest_frame_index(frames, int(beat)))

    seeds = sorted(set(seeds))
    selected: List[Dict[str, Any]] = []
    previous_frame = None

    def movement_start_index(key_idx: int) -> int:
        """Find the settled pose immediately before the movement shown by a cue."""
        key_t = int(frames[key_idx].get("t_ms", 0))
        candidates = []
        for i in range(key_idx - 1, -1, -1):
            dt = key_t - int(frames[i].get("t_ms", 0))
            if dt < 260:
                continue
            if dt > 820:
                break
            candidates.append(i)
        if not candidates:
            return max(0, key_idx - 1)
        # A quiet frame gives a much clearer start -> destination gesture than
        # measuring from an arbitrary frame 320 ms earlier.
        return min(candidates, key=lambda i: motion[i])

    for seed_idx in seeds:
        key_idx = _destination_keyframe(frames, motion, seed_idx, previous_frame)
        t_ms = int(frames[key_idx].get("t_ms", 0))
        if t_ms < 450:
            continue

        novelty = 1.0 if previous_frame is None else _pose_novelty(previous_frame, frames[key_idx])
        if selected:
            gap = t_ms - int(selected[-1]["t_ms"])
            if gap < min_gap_ms:
                # Within the same phrase keep only the more distinctive pose.
                if novelty > float(selected[-1].get("novelty", 0.0)) * 1.18:
                    selected[-1] = {
                        "t_ms": t_ms,
                        "frame_index": key_idx,
                        "start_frame_index": movement_start_index(key_idx),
                        "motion": round(float(motion[seed_idx]), 5),
                        "novelty": round(float(novelty), 5),
                        "kind": "pictogram",
                    }
                    previous_frame = frames[key_idx]
                continue

        if selected and novelty < 0.055 and t_ms - int(selected[-1]["t_ms"]) < max_gap_ms:
            continue

        selected.append({
            "t_ms": t_ms,
            "frame_index": key_idx,
            "start_frame_index": movement_start_index(key_idx),
            "motion": round(float(motion[seed_idx]), 5),
            "novelty": round(float(novelty), 5),
            "kind": "pictogram",
        })
        previous_frame = frames[key_idx]

    end_ms = int(frames[-1].get("t_ms", 0))
    if not selected:
        idx = _nearest_frame_index(frames, min(1000, end_ms))
        selected.append({
            "t_ms": int(frames[idx].get("t_ms", 0)),
            "frame_index": idx,
            "start_frame_index": movement_start_index(idx),
            "motion": 0.0,
            "novelty": 0.0,
            "kind": "pictogram",
        })

    # Fill long gaps using the most visually different destination pose in the
    # missing interval, never an arbitrary interpolated frame.
    output: List[Dict[str, Any]] = [selected[0]]
    for marker in selected[1:]:
        while int(marker["t_ms"]) - int(output[-1]["t_ms"]) > max_gap_ms:
            target_t = int(output[-1]["t_ms"]) + max_gap_ms
            seed_idx = _nearest_frame_index(frames, target_t - 300)
            key_idx = _destination_keyframe(frames, motion, seed_idx, frames[int(output[-1]["frame_index"])])
            key_t = int(frames[key_idx].get("t_ms", target_t))
            if key_t <= int(output[-1]["t_ms"]) + min_gap_ms * 0.65:
                key_t = min(target_t, end_ms)
                key_idx = _nearest_frame_index(frames, key_t)
            output.append({
                "t_ms": int(frames[key_idx].get("t_ms", key_t)),
                "frame_index": key_idx,
                "start_frame_index": movement_start_index(key_idx),
                "motion": round(float(motion[min(seed_idx, len(motion)-1)]), 5),
                "novelty": round(float(_pose_novelty(
                    frames[int(output[-1]["frame_index"])],
                    frames[key_idx],
                )), 5),
                "kind": "pictogram_fill",
            })
        output.append(marker)

    deduped: List[Dict[str, Any]] = []
    for marker in sorted(output, key=lambda m: int(m["t_ms"])):
        if deduped and int(marker["t_ms"]) - int(deduped[-1]["t_ms"]) < min_gap_ms:
            continue
        deduped.append(marker)

    for i, marker in enumerate(deduped):
        marker["cue_index"] = i
    return deduped
