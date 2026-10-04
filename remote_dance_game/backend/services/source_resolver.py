import os
import re
from pathlib import Path
from typing import Callable, Optional, Tuple


SUPPORTED_URL = re.compile(r"^https?://", re.IGNORECASE)


def resolve_source(
    source_type: str,
    file_path: Optional[str],
    source_url: Optional[str],
    work_dir: str,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Tuple[str, str]:
    """Return a local video path and a display title.

    URL mode intentionally delegates provider handling to yt-dlp. That gives one
    implementation for YouTube, TikTok and other supported public video pages.
    """
    if source_type == "file":
        if not file_path or not os.path.isfile(file_path):
            raise ValueError(f"Video file not found: {file_path or ''}")
        return os.path.abspath(file_path), Path(file_path).stem

    if source_type not in {"url", "youtube", "tiktok"}:
        raise ValueError(f"Unsupported source_type: {source_type}")
    if not source_url or not SUPPORTED_URL.match(source_url):
        raise ValueError("A valid http(s) video URL is required")

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed; run pip install -r requirements.txt") from exc

    os.makedirs(work_dir, exist_ok=True)
    output_template = os.path.join(work_dir, "source.%(ext)s")

    last_progress = -1

    def hook(data):
        nonlocal last_progress
        if not progress_cb or data.get("status") != "downloading":
            return
        total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
        downloaded = data.get("downloaded_bytes") or 0
        if total:
            pct = max(0, min(100, int(downloaded * 100 / total)))
            if pct >= last_progress + 5:
                last_progress = pct
                progress_cb(pct, "downloading_source")

    options = {
        "outtmpl": output_template,
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "socket_timeout": 20,
        "retries": 3,
        "fragment_retries": 3,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(source_url, download=True)
        title = str(info.get("title") or "Imported dance")
        requested = info.get("requested_downloads") or []
        candidates = []
        if requested:
            candidates.extend(d.get("filepath") for d in requested if d.get("filepath"))
        prepared = ydl.prepare_filename(info)
        candidates.extend([
            prepared,
            os.path.splitext(prepared)[0] + ".mp4",
            os.path.join(work_dir, "source.mp4"),
        ])

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate), title

    matches = sorted(Path(work_dir).glob("source.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for match in matches:
        if match.is_file() and match.suffix.lower() not in {".part", ".ytdl"}:
            return str(match.resolve()), title

    raise RuntimeError("The remote video was downloaded but no media file was produced")
