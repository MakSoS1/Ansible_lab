import json
import os
import subprocess
from typing import Optional


def get_ffmpeg_path() -> str:
    return os.environ.get("FFMPEG_BIN", "ffmpeg")


def get_ffprobe_path() -> str:
    return os.environ.get("FFPROBE_BIN", "ffprobe")


def _run(cmd, timeout: Optional[float] = None):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "media command failed").strip()
        raise RuntimeError(message[-2000:])
    return result


def probe_video(file_path: str) -> dict:
    if not os.path.isfile(file_path):
        raise ValueError(f"Video not found: {file_path}")
    result = _run([
        get_ffprobe_path(), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", file_path,
    ], timeout=30)
    data = json.loads(result.stdout or "{}")
    video_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        raise ValueError("Source does not contain a video stream")
    return data


def _clip_args(clip_start: float, clip_end: Optional[float]):
    args = []
    if clip_start > 0:
        args.extend(["-ss", str(clip_start)])
    if clip_end is not None:
        duration = clip_end - clip_start
        if duration <= 0:
            raise ValueError("Clip end must be greater than clip start")
        args.extend(["-t", str(duration)])
    return args


def normalize_video(input_path: str, output_path: str, clip_start: float = 0.0,
                    clip_end: Optional[float] = None, proxy_height: int = 720,
                    fps: int = 30) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [get_ffmpeg_path(), "-y"]
    if clip_start > 0:
        cmd.extend(["-ss", str(clip_start)])
    cmd.extend(["-i", input_path])
    if clip_end is not None:
        cmd.extend(["-t", str(clip_end - clip_start)])
    cmd.extend([
        "-vf", f"scale=-2:{proxy_height},fps={fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-an", output_path,
    ])
    _run(cmd, timeout=None)
    return output_path


def extract_audio(input_path: str, output_path: str, clip_start: float = 0.0,
                  clip_end: Optional[float] = None) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [get_ffmpeg_path(), "-y"]
    if clip_start > 0:
        cmd.extend(["-ss", str(clip_start)])
    cmd.extend(["-i", input_path])
    if clip_end is not None:
        cmd.extend(["-t", str(clip_end - clip_start)])
    cmd.extend(["-vn", "-acodec", "libmp3lame", "-q:a", "2", output_path])
    _run(cmd, timeout=None)
    return output_path


def create_preview_video(input_path: str, output_path: str, clip_start: float = 0.0,
                         clip_end: Optional[float] = None, width: int = 1920,
                         height: int = 1080, fps: int = 30) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [get_ffmpeg_path(), "-y"]
    if clip_start > 0:
        cmd.extend(["-ss", str(clip_start)])
    cmd.extend(["-i", input_path])
    if clip_end is not None:
        cmd.extend(["-t", str(clip_end - clip_start)])
    cmd.extend([
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output_path,
    ])
    _run(cmd, timeout=None)
    return output_path
