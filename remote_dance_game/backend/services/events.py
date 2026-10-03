import numpy as np
from typing import List, Dict, Any
from config import (
    HOLD_DETECT_WINDOW_MS, HOLD_DETECT_MIN_SIMILARITY_DELTA,
    HOLD_DETECT_MAX_MOTION, HOLD_DETECT_MIN_DURATION_MS, HOLD_CONFIG,
)


def detect_hold_events(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(frames) < 10:
        return []

    events = []
    fps = 30
    window_size = max(2, int(HOLD_DETECT_WINDOW_MS / 1000.0 * fps))

    motion_energies = []
    for f in frames:
        me = f.get("motion_energy", {})
        total = me.get("arms", 0) + me.get("legs", 0) + me.get("torso", 0)
        motion_energies.append(total)

    smoothed_motion = []
    for i in range(len(motion_energies)):
        start = max(0, i - window_size // 2)
        end = min(len(motion_energies), i + window_size // 2 + 1)
        smoothed_motion.append(np.mean(motion_energies[start:end]))

    i = window_size
    while i < len(frames) - window_size:
        prev_avg = np.mean(smoothed_motion[max(0, i - window_size):i])
        curr_avg = smoothed_motion[i]

        if prev_avg > HOLD_DETECT_MAX_MOTION * 3 and curr_avg < HOLD_DETECT_MAX_MOTION:
            start_hold = i
            while start_hold > 0 and smoothed_motion[start_hold] > HOLD_DETECT_MAX_MOTION:
                start_hold -= 1

            end_hold = i
            while (end_hold < len(frames) - 1 and
                   smoothed_motion[end_hold] < HOLD_DETECT_MAX_MOTION * 2):
                end_hold += 1

            hold_duration_ms = frames[end_hold]["t_ms"] - frames[start_hold]["t_ms"]

            if hold_duration_ms >= HOLD_DETECT_MIN_DURATION_MS:
                best_frame = start_hold
                best_v = 0
                for fi in range(max(0, start_hold - 5), start_hold + 1):
                    lm = frames[fi].get("landmarks", [])
                    if lm and len(lm) >= 33:
                        avg_v = np.mean([l.get("v", 0) for l in lm[:33]])
                        if avg_v > best_v:
                            best_v = avg_v
                            best_frame = fi

                events.append({
                    "id": f"evt_{len(events):04d}",
                    "type": "hold",
                    "start_ms": frames[start_hold]["t_ms"],
                    "end_ms": frames[end_hold]["t_ms"],
                    "target_frame_index": best_frame,
                    "min_similarity": HOLD_CONFIG["min_similarity"],
                    "max_motion_energy": HOLD_CONFIG["max_motion_energy"],
                    "bonus_score": HOLD_CONFIG["bonus_score"],
                    "ui_text": "YEAH",
                })

            i = end_hold + window_size
        else:
            i += 1

    return events
