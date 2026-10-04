import cv2
import os
import urllib.request
from typing import Dict, Any, Optional, Callable
from config import POSE_EXTRACTION_FPS, CACHE_DIR

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_FILE = "pose_landmarker_heavy.task"
LEFT_RIGHT_SWAP_PAIRS = [
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def _get_model_path() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_path = os.path.join(CACHE_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"Downloading pose model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Model downloaded.")
    return model_path


def extract_poses_from_video(video_path: str, fps: int = POSE_EXTRACTION_FPS,
                             progress_cb: Optional[Callable] = None,
                             mirror_mode: bool = True) -> Dict[str, Any]:
    import mediapipe as mp

    model_path = _get_model_path()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_ms = int(total_frames / video_fps * 1000)

    if progress_cb:
        progress_cb(0, "pose_extraction")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )

    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    frames_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_ms = int(frame_idx / video_fps * 1000)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, t_ms)

        frame_entry = {
            "t_ms": t_ms,
            "landmarks": [],
            "world_landmarks": [],
        }

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            for lm in result.pose_landmarks[0]:
                frame_entry["landmarks"].append({
                    "x": round(float(lm.x), 4),
                    "y": round(float(lm.y), 4),
                    "z": round(float(lm.z), 4),
                    "v": round(float(getattr(lm, "visibility", getattr(lm, "presence", 1.0))), 4),
                })

            if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                for lm in result.pose_world_landmarks[0]:
                    frame_entry["world_landmarks"].append({
                        "x": round(float(lm.x), 4),
                        "y": round(float(lm.y), 4),
                        "z": round(float(lm.z), 4),
                        "v": round(float(getattr(lm, "visibility", getattr(lm, "presence", 1.0))), 4),
                    })

            if len(frame_entry["landmarks"]) < 33:
                frame_entry["landmarks"].extend([{"x": 0.0, "y": 0.0, "z": 0.0, "v": 0.0}] * (33 - len(frame_entry["landmarks"])))
            if len(frame_entry["world_landmarks"]) < 33:
                frame_entry["world_landmarks"].extend([{"x": 0.0, "y": 0.0, "z": 0.0, "v": 0.0}] * (33 - len(frame_entry["world_landmarks"])))

            if mirror_mode:
                frame_entry = mirror_landmarks(frame_entry)
        else:
            frame_entry["landmarks"] = [{"x": 0.0, "y": 0.0, "z": 0.0, "v": 0.0}] * 33
            frame_entry["world_landmarks"] = [{"x": 0.0, "y": 0.0, "z": 0.0, "v": 0.0}] * 33

        frames_data.append(frame_entry)
        frame_idx += 1

        if progress_cb and frame_idx % 30 == 0:
            pct = int(frame_idx / max(total_frames, 1) * 100)
            progress_cb(min(pct, 99), "pose_extraction")

    cap.release()
    landmarker.close()

    if progress_cb:
        progress_cb(100, "pose_extraction")

    return {
        "fps": int(video_fps),
        "total_frames": frame_idx,
        "duration_ms": duration_ms,
        "frames": frames_data,
    }


def mirror_landmarks(frame_entry: dict) -> dict:
    lm = frame_entry.get("landmarks", [])
    if len(lm) < 33:
        return frame_entry

    mirrored_lm = [dict(l) for l in lm[:33]]
    for l in mirrored_lm:
        l["x"] = round(1.0 - float(l["x"]), 4)
    for i, j in LEFT_RIGHT_SWAP_PAIRS:
        mirrored_lm[i], mirrored_lm[j] = mirrored_lm[j], mirrored_lm[i]

    world_lm = frame_entry.get("world_landmarks", [])
    mirrored_world = []
    if len(world_lm) >= 33:
        mirrored_world = [dict(l) for l in world_lm[:33]]
        for l in mirrored_world:
            l["x"] = round(-float(l["x"]), 4)
        for i, j in LEFT_RIGHT_SWAP_PAIRS:
            mirrored_world[i], mirrored_world[j] = mirrored_world[j], mirrored_world[i]

    return {
        "t_ms": frame_entry["t_ms"],
        "landmarks": mirrored_lm,
        "world_landmarks": mirrored_world,
    }
