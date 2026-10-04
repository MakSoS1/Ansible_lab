import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from config import HOLD_CONFIG, SCORE_POINTS, SMOOTHING_ALPHA
from models.schemas import GradeEnum, HoldEventResult, ScoreEvent
from services.player_features import compute_angles, compute_motion_energy, compute_velocities, visibility_score
from services.pose_comparator import compare_poses
from services.pose_normalizer import normalize_frame_for_scoring
from services.move_events import build_move_markers
from services.weights import get_weights_at_time


class HoldStateMachine:
    IDLE = "idle"
    ENTERING = "entering"
    HOLDING = "holding"
    SUCCESS = "success"
    FAIL = "fail"

    def __init__(self, event: Dict[str, Any]):
        self.event = event
        self.state = self.IDLE
        self.enter_time_ms: Optional[int] = None
        self.hold_start_ms: Optional[int] = None
        self.last_sample_ms: Optional[int] = None
        self.hold_frames = 0
        self.similarity_sum = 0.0
        self.stillness_sum = 0.0

    def update(self, t_ms: int, similarity: float, motion_energy: float) -> Tuple[str, Optional[HoldEventResult]]:
        if t_ms < self.event["start_ms"]:
            return self.state, None
        if t_ms > self.event["end_ms"]:
            if self.state in [self.HOLDING, self.ENTERING]:
                return self._finalize(t_ms)
            return self.state, None

        self.last_sample_ms = t_ms
        if self.state == self.IDLE:
            if similarity >= self.event.get("min_similarity", HOLD_CONFIG["min_similarity"]):
                self.state = self.ENTERING
                self.enter_time_ms = t_ms
                self.similarity_sum = similarity
                self.stillness_sum = 1.0 - min(motion_energy, 1.0)
                self.hold_frames = 1
        elif self.state == self.ENTERING:
            self.similarity_sum += similarity
            self.stillness_sum += 1.0 - min(motion_energy, 1.0)
            self.hold_frames += 1
            if similarity >= self.event.get("min_similarity", HOLD_CONFIG["min_similarity"]):
                if self.enter_time_ms is not None and t_ms - self.enter_time_ms >= HOLD_CONFIG["min_hold_ms"]:
                    self.state = self.HOLDING
                    self.hold_start_ms = self.enter_time_ms
            else:
                self.state = self.IDLE
                self.hold_frames = 0
        elif self.state == self.HOLDING:
            self.similarity_sum += similarity
            self.stillness_sum += 1.0 - min(motion_energy, 1.0)
            self.hold_frames += 1
            if similarity < self.event.get("min_similarity", HOLD_CONFIG["min_similarity"]) * 0.8:
                if motion_energy > self.event.get("max_motion_energy", HOLD_CONFIG["max_motion_energy"]) * 3:
                    return self._finalize(t_ms)
        return self.state, None

    def _finalize(self, t_ms: int) -> Tuple[str, Optional[HoldEventResult]]:
        if self.hold_frames < 1:
            self.state = self.FAIL
            return self.state, HoldEventResult(event_id=self.event["id"], success=False, hold_score=0, coverage=0, yeah=False)

        avg_sim = self.similarity_sum / max(self.hold_frames, 1)
        avg_still = self.stillness_sum / max(self.hold_frames, 1)
        hold_score = 0.75 * avg_sim + 0.25 * avg_still
        event_duration = self.event["end_ms"] - self.event["start_ms"]
        hold_start_ms = self.hold_start_ms if self.hold_start_ms is not None else self.enter_time_ms
        hold_end_ms = self.last_sample_ms if self.last_sample_ms is not None else t_ms
        hold_duration_ms = max(0, hold_end_ms - (hold_start_ms if hold_start_ms is not None else hold_end_ms))
        coverage = min(1.0, hold_duration_ms / max(event_duration, 1))
        success = coverage >= HOLD_CONFIG["min_hold_coverage"]
        yeah = success and hold_score >= HOLD_CONFIG["min_similarity"]
        self.state = self.SUCCESS if success else self.FAIL
        return self.state, HoldEventResult(
            event_id=self.event["id"], success=success, hold_score=round(hold_score, 3), coverage=round(coverage, 3), yeah=yeah
        )


class ScoringEngine:
    """Low-latency scorer driven by the desktop media clock.

    Player features are derived server-side from normalized pose history, so the
    phone only needs to send MediaPipe landmarks. This keeps the mobile payload
    small and prevents the old 0.8425 ceiling caused by missing angles/velocity.
    """

    def __init__(self, dance_data: Dict[str, Any]):
        self.dance_data = dance_data
        self.ref_frames = dance_data.get("reference_pose", {}).get("frames", [])
        self.weights_list = dance_data.get("weights", [])
        self.events = dance_data.get("events", {}).get("events", [])
        self.duration_ms = int(dance_data.get("duration_ms", 0) or 0)
        self.timing = dance_data.get("timing", {}) or {}

        self.ref_norm_cache: Dict[int, Dict[str, Any]] = {}
        self._norm_cache_built = False
        self.pose_buffer = deque(maxlen=90)
        self._previous_player_norm: Optional[Dict[str, Any]] = None
        self._previous_player_capture_ms: Optional[int] = None

        self.smoothed_score = 0.5
        self.combo = 0
        self.max_combo = 0
        self.total_score = 0
        self.grade_counts = {"perfect": 0, "super": 0, "good": 0, "ok": 0, "x": 0}
        self.hold_state_machines = [HoldStateMachine(e) for e in self.events if e.get("type") == "hold"]
        self.hold_results: List[Dict[str, Any]] = []
        self.timeline_scores: List[Dict[str, Any]] = []
        self.part_accuracy = {"arms": [], "legs": [], "torso": []}
        self.last_score_time_ms = -10_000
        self.game_start_time: Optional[float] = None
        self.is_running = False
        self.is_paused = False
        self._has_scored = False
        self.move_markers: List[Dict[str, Any]] = []
        self._last_awarded_move = -1

    @staticmethod
    def _decorate_features(norm: Dict[str, Any], previous: Optional[Dict[str, Any]], dt_seconds: float) -> Dict[str, Any]:
        landmarks = norm.get("landmarks", [])
        velocities = compute_velocities(landmarks, previous.get("landmarks", []) if previous else None, dt_seconds)
        norm["angles"] = compute_angles(landmarks)
        norm["velocities"] = velocities
        norm["motion_energy"] = compute_motion_energy(velocities)
        norm["tracking_score"] = visibility_score(landmarks)
        return norm

    def _precompute_ref_norm(self):
        if self._norm_cache_built:
            return
        self._norm_cache_built = True
        previous: Optional[Dict[str, Any]] = None
        previous_t: Optional[int] = None
        for i, frame in enumerate(self.ref_frames):
            norm = normalize_frame_for_scoring(frame)
            if not norm.get("valid"):
                continue
            t_ms = int(frame.get("t_ms", i * 33))
            dt = max(1e-3, (t_ms - previous_t) / 1000.0) if previous_t is not None else 0.0
            norm = self._decorate_features(norm, previous, dt)
            norm["t_ms"] = t_ms
            self.ref_norm_cache[i] = norm
            previous = norm
            previous_t = t_ms

    def start(self):
        self._precompute_ref_norm()
        self.game_start_time = time.time()
        self.is_running = True
        self.is_paused = False
        self.combo = 0
        self.max_combo = 0
        self.total_score = 0
        self.grade_counts = {"perfect": 0, "super": 0, "good": 0, "ok": 0, "x": 0}
        self.hold_results = []
        self.timeline_scores = []
        self.part_accuracy = {"arms": [], "legs": [], "torso": []}
        self.hold_state_machines = [HoldStateMachine(e) for e in self.events if e.get("type") == "hold"]
        self.smoothed_score = 0.5
        self._has_scored = False
        self.move_markers = build_move_markers(self.ref_frames, self.timing)
        self._last_awarded_move = -1
        self.last_score_time_ms = -10_000
        self.pose_buffer.clear()
        self._previous_player_norm = None
        self._previous_player_capture_ms = None

    def stop(self):
        self.is_running = False
        self.is_paused = False

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def add_pose_frame(self, pose_frame: Dict[str, Any]):
        norm = normalize_frame_for_scoring(pose_frame)
        if not norm.get("valid"):
            return
        capture_ms = int(pose_frame.get("timestamp_ms") or pose_frame.get("t_ms") or int(time.time() * 1000))
        dt = 0.0
        if self._previous_player_capture_ms is not None:
            dt = (capture_ms - self._previous_player_capture_ms) / 1000.0
        norm = self._decorate_features(norm, self._previous_player_norm, dt)
        norm["capture_ms"] = capture_ms
        norm["received_at_ms"] = int(pose_frame.get("received_at_ms") or int(time.time() * 1000))
        self.pose_buffer.append(norm)
        self._previous_player_norm = norm
        self._previous_player_capture_ms = capture_ms

    def _candidate_ref_indices(self, t_ms: int) -> List[int]:
        if not self.ref_frames:
            return []
        window_ms = int(config.MATCHING_WINDOW_MS)
        search_range = int(window_ms / 33.3) + 3
        center_idx = int(t_ms / 33.3)
        start_idx = max(0, center_idx - search_range)
        end_idx = min(len(self.ref_frames) - 1, center_idx + search_range)
        return [i for i in range(start_idx, end_idx + 1) if i in self.ref_norm_cache and abs(int(self.ref_frames[i]["t_ms"]) - t_ms) <= window_ms]

    def _find_best_ref_frame(self, t_ms: int, player_frame: Dict[str, Any], body_w: Dict[str, float]) -> Optional[Tuple[int, Dict[str, Any], float]]:
        best: Optional[Tuple[int, Dict[str, Any], float]] = None
        best_rank = -1.0
        for idx in self._candidate_ref_indices(t_ms):
            ref_norm = self.ref_norm_cache[idx]
            pose_only, _ = compare_poses(ref_norm, player_frame, 0.0, body_w, include_timing=False)
            diff = abs(int(self.ref_frames[idx]["t_ms"]) - t_ms)
            rank = pose_only - (diff / max(float(config.MATCHING_WINDOW_MS), 1.0)) * 0.03
            if rank > best_rank:
                best_rank = rank
                best = (idx, ref_norm, float(diff))
        return best

    def _move_for_time(self, t_ms: int) -> Tuple[int, Optional[Dict[str, Any]], bool]:
        if not self.move_markers:
            return -1, None, False
        # Prefer the next choreography marker until we pass its late tolerance.
        next_idx = min(self._last_awarded_move + 1, len(self.move_markers) - 1)
        marker = self.move_markers[next_idx]
        early = int(config.MOVE_SCORE_EARLY_TOLERANCE_MS)
        late = int(config.MOVE_SCORE_LATE_TOLERANCE_MS)
        due = next_idx > self._last_awarded_move and (marker["t_ms"] - early) <= t_ms <= (marker["t_ms"] + late)
        # If a media-clock jump skipped the small window, grade the missed move on
        # the next tick instead of silently dropping player feedback.
        if next_idx > self._last_awarded_move and t_ms > marker["t_ms"] + late:
            due = True
        return next_idx, marker, due

    def _select_player_frame(self, now_server_ms: int, ref_norm: Dict[str, Any], body_w: Dict[str, float]) -> Optional[Dict[str, Any]]:
        if not self.pose_buffer:
            return None
        recent_window = int(config.PLAYER_RECENT_WINDOW_MS)
        best = None
        best_rank = -1.0
        for frame in reversed(self.pose_buffer):
            age = max(0, now_server_ms - int(frame.get("received_at_ms", now_server_ms)))
            if age > recent_window:
                break
            tracking = float(frame.get("tracking_score", 0.0))
            if tracking < 0.25:
                continue
            pose_only, _ = compare_poses(ref_norm, frame, 0.0, body_w, include_timing=False)
            recency_penalty = (age / max(float(recent_window), 1.0)) * 0.035
            rank = pose_only + min(tracking, 1.0) * 0.025 - recency_penalty
            if rank > best_rank:
                best_rank = rank
                best = frame
        return best or self.pose_buffer[-1]

    def _tracking_event(self, t_ms: int, age_ms: Optional[int]) -> ScoreEvent:
        self.combo = 0
        return ScoreEvent(
            timestamp_ms=t_ms,
            grade=GradeEnum.x,
            score=0,
            total_score=self.total_score,
            combo=0,
            similarity=0.0,
            tracking_lost=True,
            pose_age_ms=age_ms,
        )

    def score_tick(self, t_ms: int, now_server_ms: Optional[int] = None) -> Optional[ScoreEvent]:
        if not self.is_running or self.is_paused:
            return None
        if t_ms - self.last_score_time_ms < int(config.SCORING_EVENT_INTERVAL_MS):
            return None
        self.last_score_time_ms = t_ms

        move_index, marker, is_move_grade = self._move_for_time(t_ms)
        target_t_ms = int(marker["t_ms"]) if marker and is_move_grade else t_ms

        if not self.pose_buffer:
            event = self._tracking_event(target_t_ms, None)
            event.move_index = move_index
            event.move_count = len(self.move_markers)
            event.is_move_grade = bool(is_move_grade)
            if is_move_grade:
                self._last_awarded_move = move_index
            return event

        now_server_ms = int(now_server_ms or time.time() * 1000)
        body_w = get_weights_at_time(self.weights_list, target_t_ms)

        # First find the relevant reference pose, then choose the best recent
        # high-confidence player frame. This is more tolerant to one bad camera
        # inference or a few milliseconds of Wi-Fi jitter without relaxing the
        # choreography itself.
        provisional = self.pose_buffer[-1]
        match = self._find_best_ref_frame(target_t_ms, provisional, body_w)
        if match is None:
            event = self._tracking_event(target_t_ms, None)
            event.move_index = move_index
            event.move_count = len(self.move_markers)
            event.is_move_grade = bool(is_move_grade)
            if is_move_grade:
                self._last_awarded_move = move_index
            return event

        ref_idx, ref_norm, time_offset = match
        player_frame = self._select_player_frame(now_server_ms, ref_norm, body_w)
        if player_frame is None:
            return None
        age_ms = max(0, now_server_ms - int(player_frame.get("received_at_ms", now_server_ms)))
        if age_ms > int(config.PLAYER_FRAME_STALE_MS) or float(player_frame.get("tracking_score", 0.0)) < 0.25:
            event = self._tracking_event(target_t_ms, age_ms)
            event.move_index = move_index
            event.move_count = len(self.move_markers)
            event.is_move_grade = bool(is_move_grade)
            if is_move_grade:
                self._last_awarded_move = move_index
            return event

        # Re-run temporal matching with the selected player frame.
        match = self._find_best_ref_frame(target_t_ms, player_frame, body_w)
        if match is None:
            return None
        ref_idx, ref_norm, time_offset = match
        similarity, part_scores = compare_poses(ref_norm, player_frame, time_offset, body_w, include_timing=True)

        if self._has_scored:
            self.smoothed_score = SMOOTHING_ALPHA * similarity + (1.0 - SMOOTHING_ALPHA) * self.smoothed_score
        else:
            self.smoothed_score = similarity
            self._has_scored = True
        grade = self._classify_grade(self.smoothed_score)

        points = 0
        if is_move_grade:
            points = int(SCORE_POINTS.get(grade.value, 0))
            if grade != GradeEnum.x:
                self.combo += 1
                combo_factor = min(0.50, self.combo * 0.01)
                points += int(points * combo_factor)
            else:
                self.combo = 0

            self.max_combo = max(self.max_combo, self.combo)
            self.total_score += points
            self.grade_counts[grade.value] += 1
            self._last_awarded_move = move_index

            for key in ("arms", "legs", "torso"):
                if key in part_scores:
                    self.part_accuracy[key].append(part_scores[key])

            self.timeline_scores.append({
                "t_ms": target_t_ms,
                "similarity": round(similarity, 3),
                "grade": grade.value,
                "score": points,
                "combo": self.combo,
                "timing_offset_ms": int(time_offset),
                "move_index": move_index,
            })

        hold_state = None
        me = player_frame.get("motion_energy", {})
        total_me = float(me.get("arms", 0.0)) + float(me.get("legs", 0.0)) + float(me.get("torso", 0.0))
        for hsm in self.hold_state_machines:
            state, result = hsm.update(t_ms, similarity, total_me)
            if result:
                self.hold_results.append(result.dict())
                if result.yeah:
                    bonus = int(hsm.event.get("bonus_score", HOLD_CONFIG["bonus_score"]))
                    self.total_score += bonus
                    hold_state = "yeah"
                elif result.success:
                    hold_state = "hold_success"
            elif state in [HoldStateMachine.ENTERING, HoldStateMachine.HOLDING]:
                hold_state = state

        raw_ref = self.ref_frames[ref_idx].get("landmarks", []) if ref_idx < len(self.ref_frames) else []
        return ScoreEvent(
            timestamp_ms=target_t_ms,
            grade=grade,
            score=points,
            total_score=self.total_score,
            combo=self.combo,
            similarity=round(similarity, 3),
            hold_state=hold_state,
            limb_scores=part_scores,
            timing_offset_ms=int(time_offset),
            tracking_lost=False,
            pose_age_ms=age_ms,
            coach_pose=raw_ref,
            move_index=move_index,
            move_count=len(self.move_markers),
            is_move_grade=bool(is_move_grade),
        )

    def _classify_grade(self, score: float) -> GradeEnum:
        th = config.SCORING_THRESHOLDS
        if score >= th["perfect"]:
            return GradeEnum.perfect
        if score >= th["super"]:
            return GradeEnum.super
        if score >= th["good"]:
            return GradeEnum.good
        if score >= th["ok"]:
            return GradeEnum.ok
        return GradeEnum.x

    def get_final_results(self) -> Dict[str, Any]:
        acc: Dict[str, float] = {}
        for key in ("arms", "legs", "torso"):
            vals = self.part_accuracy.get(key, [])
            acc[key] = round(float(np.mean(vals)), 3) if vals else 0.0
        return {
            "total_score": self.total_score,
            "max_combo": self.max_combo,
            "grade_counts": self.grade_counts,
            "hold_results": self.hold_results,
            "accuracy_arms": acc.get("arms", 0.0),
            "accuracy_legs": acc.get("legs", 0.0),
            "accuracy_torso": acc.get("torso", 0.0),
            "timeline_scores": self.timeline_scores[-400:],
        }
