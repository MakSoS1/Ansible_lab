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


def build_pictogram_markers(
    frames: List[Dict[str, Any]],
    timing: Dict[str, Any],
    min_gap_ms: int = 950,
    max_gap_ms: int = 1750,
) -> List[Dict[str, Any]]:
    """Sparse, readable Just-Dance-style pictogram cues.

    Scoring remains dense (~0.3-0.65 s), but the visual cue rail should not.
    This selects distinctive key poses roughly once per second or slower and
    only falls back to a quiet-section cue when the gap becomes too long.
    """
    if not frames:
        return []

    dense = build_move_markers(frames, timing)
    if not dense:
        return []

    candidates = [m for m in dense if m.get("kind") == "motion_peak"]
    if len(candidates) < 3:
        candidates = dense

    selected: List[Dict[str, Any]] = []
    last_frame = None
    last_t = -100000
    for marker in candidates:
        t_ms = int(marker["t_ms"])
        if t_ms < 450:
            continue
        idx = int(marker["frame_index"])
        frame = frames[idx]
        novelty = 1.0 if last_frame is None else _pose_novelty(last_frame, frame)
        gap = t_ms - last_t

        # A key pose should either be clearly different or have enough temporal
        # distance to be useful as a visual instruction.
        if not selected or (
            gap >= min_gap_ms
            and (novelty >= 0.085 or gap >= max_gap_ms)
        ):
            item = dict(marker)
            item["novelty"] = round(float(novelty), 5)
            item["kind"] = "pictogram"
            selected.append(item)
            last_frame = frame
            last_t = t_ms

    end_ms = int(frames[-1].get("t_ms", 0))
    if not selected:
        idx = _nearest_frame_index(frames, min(900, end_ms))
        selected.append({
            "t_ms": int(frames[idx].get("t_ms", 0)),
            "frame_index": idx,
            "motion": 0.0,
            "novelty": 0.0,
            "kind": "pictogram",
        })

    # Fill only genuinely long holes, choosing the most dynamic dense marker in
    # the interval rather than a random interpolated pose.
    output: List[Dict[str, Any]] = []
    for marker in selected:
        if output and marker["t_ms"] - output[-1]["t_ms"] > max_gap_ms:
            left = int(output[-1]["t_ms"]) + min_gap_ms
            right = int(marker["t_ms"]) - min_gap_ms // 2
            between = [m for m in dense if left <= int(m["t_ms"]) <= right]
            if between:
                filler = max(between, key=lambda m: float(m.get("motion", 0.0)))
                filler = dict(filler)
                filler["kind"] = "pictogram_fill"
                filler["novelty"] = round(_pose_novelty(
                    frames[int(output[-1]["frame_index"])],
                    frames[int(filler["frame_index"])],
                ), 5)
                output.append(filler)
        output.append(marker)

    # Deduplicate after fill and assign a cue index.
    deduped: List[Dict[str, Any]] = []
    for marker in sorted(output, key=lambda m: int(m["t_ms"])):
        if deduped and int(marker["t_ms"]) - int(deduped[-1]["t_ms"]) < min_gap_ms * 0.65:
            if float(marker.get("motion", 0.0)) > float(deduped[-1].get("motion", 0.0)):
                deduped[-1] = marker
            continue
        deduped.append(marker)

    for i, marker in enumerate(deduped):
        marker["cue_index"] = i
    return deduped
