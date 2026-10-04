from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from services.video_processor import create_game_master, create_preview_video, normalize_video, probe_video


def run() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "source.mp4"
        normalized = root / "normalized.mp4"
        preview = root / "preview.mp4"
        master = root / "game_master.mp4"
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
            "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        meta = probe_video(str(source))
        if not meta.get("streams"):
            raise RuntimeError("ffprobe returned no streams")

        normalize_video(str(source), str(normalized), 0.25, 1.5)
        create_preview_video(str(source), str(preview), 0.25, 1.5)
        master_meta = create_game_master(
            str(source), str(master), 0.25, 1.5,
            target_height=1080, crf=20, preset="veryfast",
        )

        for path in (normalized, preview, master):
            if not path.exists() or path.stat().st_size < 10_000:
                raise RuntimeError(f"media output is invalid: {path}")

        master_probe = probe_video(str(master))
        video = next(s for s in master_probe["streams"] if s.get("codec_type") == "video")
        if int(video.get("width", 0)) != 1920 or int(video.get("height", 0)) != 1080:
            raise RuntimeError(f"unexpected game master resolution: {video.get('width')}x{video.get('height')}")
        if not any(s.get("codec_type") == "audio" for s in master_probe["streams"]):
            raise RuntimeError("game master lost its audio stream")
        if master_meta["mode"] != "video_first_hq":
            raise RuntimeError(f"wrong master mode: {master_meta}")

        print("media smoke: OK", master_meta)


if __name__ == "__main__":
    run()
