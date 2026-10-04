from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ["DANCE_DISABLE_YOLO"] = "1"

from services.stylized_video import render_game_video
from services.video_processor import probe_video


def make_fixture(path: Path, fps: int = 18, duration: float = 2.0):
    width, height = 480, 270
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frames = []
    for i in range(int(fps * duration)):
        t = i / fps
        image = np.zeros((height, width, 3), np.uint8)
        image[:] = (28, 24, 18)
        cx, cy = int(width * (.5 + .08 * math.sin(t * 2.1))), int(height * .46)
        pose = [{"x": 0.0, "y": 0.0, "z": 0.0, "v": 0.0} for _ in range(33)]
        def p(idx, x, y): pose[idx] = {"x": x / width, "y": y / height, "z": 0.0, "v": 1.0}
        p(0, cx, cy - 62); p(11, cx - 21, cy - 34); p(12, cx + 21, cy - 34)
        swing = int(math.sin(t * 4.0) * 34)
        p(13, cx - 43, cy - 12 - swing); p(15, cx - 67, cy - swing)
        p(14, cx + 43, cy - 12 + swing); p(16, cx + 67, cy + swing)
        p(23, cx - 15, cy + 16); p(24, cx + 15, cy + 16)
        p(25, cx - 23, cy + 58); p(27, cx - 28, cy + 101)
        p(26, cx + 23, cy + 58); p(28, cx + 28, cy + 101)
        for a, b in ((11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)):
            pa, pb = pose[a], pose[b]
            cv2.line(image, (int(pa["x"]*width), int(pa["y"]*height)), (int(pb["x"]*width), int(pb["y"]*height)), (85,155,225), 15, cv2.LINE_AA)
        cv2.circle(image, (cx, cy - 62), 17, (180,195,220), -1, cv2.LINE_AA)
        writer.write(image)
        frames.append({"t_ms": int(t*1000), "landmarks": pose, "world_landmarks": pose})
    writer.release()
    return {"fps": fps, "duration_ms": int(duration*1000), "total_frames": len(frames), "frames": frames}


def run():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source, audio, output, poster = root/"source.mp4", root/"audio.m4a", root/"game.mp4", root/"poster.jpg"
        poses = make_fixture(source)
        subprocess.run(["ffmpeg","-y","-f","lavfi","-i","sine=frequency=330:sample_rate=44100","-t","2","-c:a","aac",str(audio)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        timing = {"tempo":120,"beat_ms":[0,500,1000,1500],"strong_beat_ms":[0,1000]}
        meta = render_game_video(str(source), str(output), poses, timing, str(audio), "CI Neon", str(poster), False, 640, 360, 18)
        assert output.exists() and output.stat().st_size > 20_000
        assert poster.exists() and poster.stat().st_size > 4_000
        assert meta["theme"]["name"]
        streams = probe_video(str(output)).get("streams", [])
        assert any(s.get("codec_type") == "video" for s in streams)
        assert any(s.get("codec_type") == "audio" for s in streams)
        print("stylizer smoke: OK", meta["theme"]["name"], meta["segmentation_backend"])


if __name__ == "__main__":
    run()
