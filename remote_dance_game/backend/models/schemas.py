from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    file = "file"
    url = "url"
    youtube = "youtube"
    tiktok = "tiktok"


class DanceCreateRequest(BaseModel):
    source_type: SourceType = SourceType.file
    file_path: Optional[str] = None
    youtube_url: Optional[str] = None
    source_url: Optional[str] = None
    clip_start_sec: float = 0.0
    clip_end_sec: Optional[float] = None
    mirror_mode: bool = True
    title: Optional[str] = None
    difficulty: str = "medium"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    stage: str = ""
    error: Optional[str] = None
    dance_id: Optional[str] = None


class DanceListItem(BaseModel):
    dance_id: str
    title: str
    duration_ms: int
    difficulty: str
    created_at: str
    has_video: bool = False
    has_poster: bool = False
    preview_mode: str = "local_video"
    theme: Dict[str, Any] = Field(default_factory=dict)


class DanceDetailResponse(BaseModel):
    dance_id: str
    title: str
    version: int
    duration_ms: int
    skeleton_format: str
    preview_mode: str
    difficulty: str
    mirror_mode: bool
    created_at: str
    num_frames: int = 0
    num_events: int = 0
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    has_poster: bool = False
    theme: Dict[str, Any] = Field(default_factory=dict)


class Landmark(BaseModel):
    x: float
    y: float
    z: float = 0.0
    v: float = 1.0


class PoseFrame(BaseModel):
    type: str = "pose_frame"
    session_id: str = ""
    seq: int = 0
    timestamp_ms: int = 0
    landmarks: List[Landmark] = Field(default_factory=list)
    world_landmarks: List[Landmark] = Field(default_factory=list)
    tracking_score: float = 0.0
    device_rotation: float = 0.0


class GradeEnum(str, Enum):
    perfect = "perfect"
    super = "super"
    good = "good"
    ok = "ok"
    x = "x"


class ScoreEvent(BaseModel):
    type: str = "score_event"
    timestamp_ms: int = 0
    grade: GradeEnum = GradeEnum.x
    score: int = 0
    total_score: int = 0
    combo: int = 0
    similarity: float = 0.0
    hold_state: Optional[str] = None
    limb_scores: Optional[Dict[str, float]] = None
    timing_offset_ms: Optional[int] = None
    tracking_lost: bool = False
    pose_age_ms: Optional[int] = None
    coach_pose: Optional[List[Dict[str, float]]] = None
    move_index: int = -1
    move_count: int = 0
    is_move_grade: bool = False


class HoldEventResult(BaseModel):
    event_id: str
    success: bool = False
    hold_score: float = 0.0
    coverage: float = 0.0
    yeah: bool = False


class CalibrationCheck(BaseModel):
    name: str
    passed: bool
    message: str


class CalibrationResult(BaseModel):
    type: str = "calibration_result"
    checks: List[CalibrationCheck] = Field(default_factory=list)
    ready: bool = False


class GameSessionCreate(BaseModel):
    dance_id: str


class GameSessionResponse(BaseModel):
    session_id: str
    dance_id: str
    connect_url: str
    qr_data: str


class GameControlMessage(BaseModel):
    type: str
    session_id: str = ""
    action: str = ""
    data: Optional[Dict[str, Any]] = None


class ResultStat(BaseModel):
    total_score: int = 0
    max_combo: int = 0
    grade_counts: Dict[str, int] = Field(default_factory=dict)
    hold_results: List[HoldEventResult] = Field(default_factory=list)
    accuracy_arms: float = 0.0
    accuracy_legs: float = 0.0
    accuracy_torso: float = 0.0
    timeline_scores: List[Dict[str, Any]] = Field(default_factory=list)


class SettingsUpdate(BaseModel):
    scoring_thresholds: Optional[Dict[str, float]] = None
    sensitivity: Optional[float] = None
    mirror_mode_default: Optional[bool] = None
    debug_mode: Optional[bool] = None
