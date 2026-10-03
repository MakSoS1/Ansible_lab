import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.environ.get("DANCE_GAME_DIR", PROJECT_ROOT)
DATA_DIR = os.path.join(BASE_DIR, "data")
DANCES_DIR = os.path.join(DATA_DIR, "dances")
JOBS_DIR = os.path.join(DATA_DIR, "jobs")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")
INPUT_DIR = os.path.join(BASE_DIR, "input")
DB_PATH = os.path.join(DATA_DIR, "dance_game.db")

HOST = "0.0.0.0"
PORT = 8000
# Optional public HTTPS URL for phone browser access, for example:
# PUBLIC_BASE_URL=https://your-name.trycloudflare.com
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

SCORING_THRESHOLDS = {
    "perfect": 0.92,
    "super": 0.82,
    "good": 0.68,
    "ok": 0.50,
}

SCORE_POINTS = {
    "perfect": 300,
    "super": 250,
    "good": 200,
    "ok": 150,
    "x": 0,
}

SCORING_WEIGHTS = {
    "limb_vector": 0.40,
    "joint_angle": 0.25,
    "motion": 0.20,
    "timing": 0.15,
}

MATCHING_WINDOW_MS = 120
SMOOTHING_WINDOW_MS = 200
SCORING_TICK_HZ = 20
SCORING_EVENT_INTERVAL_MS = 250
PLAYER_FRAME_STALE_MS = 260
PLAYER_FRAME_FUTURE_TOLERANCE_MS = 80
SMOOTHING_ALPHA = 0.55

COMBO_MULTIPLIER = 0.1

HOLD_CONFIG = {
    "min_similarity": 0.88,
    "max_motion_energy": 0.12,
    "min_hold_ms": 300,
    "min_hold_coverage": 0.70,
    "bonus_score": 500,
}

POSE_EXTRACTION_FPS = 30
PROXY_RESOLUTION = 720

BODY_PART_WEIGHTS_RANGE = {
    "arms": (0.3, 1.0),
    "legs": (0.3, 1.0),
    "torso": (0.2, 0.8),
}

POSE_SMOOTHING_WINDOW = 7
POSE_SMOOTHING_ORDER = 2

CALIBRATION_THRESHOLDS = {
    "min_visibility": 0.7,
    "min_key_joints_visible": 8,
    "max_jitter": 0.02,
}

BLAZEPOSE_33 = list(range(33))

JOINT_ANGLE_DEFS = {
    "left_elbow": [11, 13, 15],
    "right_elbow": [12, 14, 16],
    "left_shoulder": [13, 11, 23],
    "right_shoulder": [14, 12, 24],
    "left_hip": [11, 23, 25],
    "right_hip": [12, 24, 26],
    "left_knee": [23, 25, 27],
    "right_knee": [24, 26, 28],
    "left_wrist_shoulder": [13, 15, 11],
    "right_wrist_shoulder": [14, 16, 12],
    "left_ankle_knee": [25, 27, 29],
    "right_ankle_knee": [26, 28, 30],
}

LIMB_VECTOR_DEFS = {
    "left_upper_arm": [11, 13],
    "left_forearm": [13, 15],
    "right_upper_arm": [12, 14],
    "right_forearm": [14, 16],
    "left_thigh": [23, 25],
    "left_shin": [25, 27],
    "right_thigh": [24, 26],
    "right_shin": [26, 28],
}

KEY_VELOCITY_JOINTS = [15, 16, 27, 28, 13, 14, 25, 26]

ARMS_JOINTS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
LEGS_JOINTS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
TORSO_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 23, 24]

WEIGHT_WINDOW_MS = 250

HOLD_DETECT_WINDOW_MS = 800
HOLD_DETECT_MIN_SIMILARITY_DELTA = 0.15
HOLD_DETECT_MAX_MOTION = 0.08
HOLD_DETECT_MIN_DURATION_MS = 600
