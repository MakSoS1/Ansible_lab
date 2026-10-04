import json
import os
import shutil
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
        raise RuntimeError(message[-3000:])
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


def _video_stream(metadata: dict) -> dict:
    return next((s for s in metadata.get("streams", []) if s.get("codec_type") == "video"), {})


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
        "-vf", f"scale=-2:{proxy_height}:flags=lanczos,fps={fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
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
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,fps={fps},setsar=1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output_path,
    ])
    _run(cmd, timeout=None)
    return output_path


def _target_dimensions(metadata: dict, target_height: int = 2160) -> tuple[int, int]:
    stream = _video_stream(metadata)
    source_w = int(stream.get("width") or 1920)
    source_h = int(stream.get("height") or 1080)
    target_height = int(max(720, min(2160, target_height)))
    if source_h > target_height:
        target_height = source_h if source_h <= 2160 else 2160
    # Unity TV output is 16:9. Letterbox unusual sources rather than crop choreography.
    return int(round(target_height * 16 / 9)), target_height


def create_game_master(input_path: str, output_path: str, clip_start: float = 0.0,
                       clip_end: Optional[float] = None, target_height: int = 2160,
                       crf: int = 17, preset: str = "slow") -> dict:
    """Create the video-first gameplay master.

    This intentionally avoids denoise/frame interpolation: both can erase hands,
    fingers and quick limb edges that matter in dance footage. Lanczos scaling and
    a very restrained unsharp pass improve 4K-TV presentation without pretending
    to reconstruct detail that is absent from the source.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metadata = probe_video(input_path)
    stream = _video_stream(metadata)
    source_w = int(stream.get("width") or 0)
    source_h = int(stream.get("height") or 0)
    width, height = _target_dimensions(metadata, target_height)

    sharpen = os.environ.get("DANCE_VIDEO_SHARPEN", "1").lower() not in {"0", "false", "off"}
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    )
    if sharpen:
        # deliberately mild: enough to counter scaling softness, not create halos
        vf += ",unsharp=5:5:0.22:3:3:0.06"

    cmd = [get_ffmpeg_path(), "-y"]
    if clip_start > 0:
        cmd.extend(["-ss", str(clip_start)])
    cmd.extend(["-i", input_path])
    if clip_end is not None:
        duration = clip_end - clip_start
        if duration <= 0:
            raise ValueError("Clip end must be greater than clip start")
        cmd.extend(["-t", str(duration)])
    cmd.extend([
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", vf,
        "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
        "-profile:v", "high", "-level:v", "5.1", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "256k",
        "-movflags", "+faststart",
        output_path,
    ])
    _run(cmd, timeout=None)

    out_meta = probe_video(output_path)
    out_stream = _video_stream(out_meta)
    return {
        "path": output_path,
        "mode": "video_first_hq",
        "source_width": source_w,
        "source_height": source_h,
        "output_width": int(out_stream.get("width") or width),
        "output_height": int(out_stream.get("height") or height),
        "upscaled": source_h > 0 and source_h < int(out_stream.get("height") or height),
        "filter": "lanczos+mild_unsharp" if sharpen else "lanczos",
        "crf": int(crf),
        "preset": preset,
    }


def extract_poster_frame(input_path: str, output_path: str, at_sec: float = 2.0,
                         width: int = 1920, height: int = 1080) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _run([
        get_ffmpeg_path(), "-y", "-ss", str(max(0.0, at_sec)), "-i", input_path,
        "-frames:v", "1",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
        "-q:v", "2", output_path,
    ], timeout=None)
    return output_path


def optional_ai_upscale(input_path: str, output_path: str) -> Optional[str]:
    """Optional external AI-SR hook.

    Set DANCE_AI_UPSCALE_CMD to a command template containing {input} and {output}.
    It is intentionally opt-in: an unavailable model must never block dance import.
    """
    template = os.environ.get("DANCE_AI_UPSCALE_CMD", "").strip()
    if not template:
        return None
    command = template.format(input=shlex_quote(input_path), output=shlex_quote(output_path))
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(output_path):
        message = (result.stderr or result.stdout or "AI upscale failed").strip()
        raise RuntimeError(message[-2000:])
    return output_path


def shlex_quote(value: str) -> str:
    if os.name == "nt":
        return '"' + value.replace('"', '\"') + '"'
    import shlex
    return shlex.quote(value)
