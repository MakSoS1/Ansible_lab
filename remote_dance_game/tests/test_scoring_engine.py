from __future__ import annotations

import copy
import math

from models.schemas import GradeEnum
from services.scoring_engine import ScoringEngine


def _base_pose() -> list[dict[str, float]]:
    # MediaPipe-like 33 landmark skeleton in normalized image coordinates.
    pts = [{"x": 0.50, "y": 0.30, "z": 0.0, "v": 1.0} for _ in range(33)]
    coords = {
        0: (0.50, 0.10), 1: (0.48, 0.09), 2: (0.47, 0.09), 3: (0.46, 0.10),
        4: (0.52, 0.09), 5: (0.53, 0.09), 6: (0.54, 0.10), 7: (0.44, 0.12),
        8: (0.56, 0.12), 9: (0.48, 0.15), 10: (0.52, 0.15),
        11: (0.40, 0.30), 12: (0.60, 0.30),
        13: (0.32, 0.44), 14: (0.68, 0.44),
        15: (0.27, 0.58), 16: (0.73, 0.58),
        17: (0.25, 0.60), 18: (0.75, 0.60), 19: (0.26, 0.59), 20: (0.74, 0.59),
        21: (0.28, 0.58), 22: (0.72, 0.58),
        23: (0.45, 0.58), 24: (0.55, 0.58),
        25: (0.43, 0.77), 26: (0.57, 0.77),
        27: (0.42, 0.95), 28: (0.58, 0.95),
        29: (0.40, 0.98), 30: (0.60, 0.98), 31: (0.39, 1.00), 32: (0.61, 1.00),
    }
    for idx, (x, y) in coords.items():
        pts[idx]["x"] = x
        pts[idx]["y"] = y
    return pts


def _pose_at(i: int) -> list[dict[str, float]]:
    pts = _base_pose()
    # Smooth, asymmetric dance motion: raise the right arm and shift the left arm.
    phase = i / 18.0 * math.pi
    raise_amount = 0.24 * math.sin(phase)
    side_amount = 0.08 * math.sin(phase * 1.3)
    pts[14]["y"] -= raise_amount * 0.45
    pts[16]["y"] -= raise_amount
    pts[16]["x"] += side_amount
    pts[13]["x"] -= side_amount * 0.5
    pts[15]["x"] -= side_amount
    return pts


def _dance() -> dict:
    frames = [{"t_ms": i * 33, "landmarks": _pose_at(i), "world_landmarks": []} for i in range(19)]
    return {
        "duration_ms": frames[-1]["t_ms"],
        "reference_pose": {"fps": 30, "frames": frames},
        "weights": [],
        "events": {"events": []},
    }


def _feed_pair(engine: ScoringEngine, previous_idx: int, current_idx: int, received_at_ms: int = 10_000) -> None:
    engine.add_pose_frame({
        "timestamp_ms": 1_000,
        "received_at_ms": received_at_ms - 10,
        "landmarks": copy.deepcopy(_pose_at(previous_idx)),
        "world_landmarks": [],
    })
    engine.add_pose_frame({
        "timestamp_ms": 1_033,
        "received_at_ms": received_at_ms,
        "landmarks": copy.deepcopy(_pose_at(current_idx)),
        "world_landmarks": [],
    })



def _first_move(engine: ScoringEngine) -> tuple[int, int]:
    assert engine.move_markers
    t_ms = int(engine.move_markers[0]["t_ms"])
    idx = max(1, min(18, round(t_ms / 33)))
    return t_ms, idx


def test_exact_phone_landmarks_can_reach_perfect() -> None:
    engine = ScoringEngine(_dance())
    engine.start()
    move_t, move_idx = _first_move(engine)
    _feed_pair(engine, move_idx - 1, move_idx)
    event = engine.score_tick(move_t, now_server_ms=10_020)
    assert event is not None
    assert event.tracking_lost is False
    assert event.grade == GradeEnum.perfect
    assert event.similarity >= 0.92
    assert event.is_move_grade is True
    assert event.score > 300  # first move grade includes the combo bonus


def test_wrong_amplitude_scores_lower_than_exact_pose() -> None:
    exact = ScoringEngine(_dance())
    exact.start()
    move_t, move_idx = _first_move(exact)
    _feed_pair(exact, move_idx - 1, move_idx)
    exact_event = exact.score_tick(move_t, now_server_ms=10_020)
    assert exact_event is not None

    wrong = ScoringEngine(_dance())
    wrong.start()
    wrong.add_pose_frame({"timestamp_ms": 1_000, "received_at_ms": 9_990, "landmarks": _pose_at(move_idx - 1)})
    altered = _pose_at(move_idx)
    # Collapse the intended arm amplitude toward the torso and put the wrist on the wrong side.
    altered[14]["x"], altered[14]["y"] = 0.58, 0.43
    altered[16]["x"], altered[16]["y"] = 0.54, 0.52
    wrong.add_pose_frame({"timestamp_ms": 1_033, "received_at_ms": 10_000, "landmarks": altered})
    wrong_event = wrong.score_tick(move_t, now_server_ms=10_020)
    assert wrong_event is not None
    assert wrong_event.similarity < exact_event.similarity - 0.05
    assert wrong_event.limb_scores is not None
    assert wrong_event.limb_scores["arms"] < exact_event.limb_scores["arms"]


def test_temporal_mismatch_is_penalized() -> None:
    exact = ScoringEngine(_dance())
    exact.start()
    move_t, move_idx = _first_move(exact)
    _feed_pair(exact, move_idx - 1, move_idx)
    exact_event = exact.score_tick(move_t, now_server_ms=10_020)
    assert exact_event is not None

    late = ScoringEngine(_dance())
    late.start()
    late_idx = max(1, move_idx - 4)
    _feed_pair(late, late_idx - 1, late_idx)
    late_event = late.score_tick(move_t, now_server_ms=10_020)
    assert late_event is not None
    assert late_event.timing_offset_ms is not None
    assert late_event.timing_offset_ms >= 90
    assert late_event.similarity < exact_event.similarity
    assert late_event.grade != GradeEnum.perfect


def test_stale_pose_awards_zero_points_and_breaks_combo() -> None:
    engine = ScoringEngine(_dance())
    engine.start()
    move_t, move_idx = _first_move(engine)
    _feed_pair(engine, move_idx - 1, move_idx, received_at_ms=1_000)
    event = engine.score_tick(move_t, now_server_ms=1_400)
    assert event is not None
    assert event.tracking_lost is True
    assert event.grade == GradeEnum.x
    assert event.score == 0
    assert event.total_score == 0
    assert event.combo == 0
