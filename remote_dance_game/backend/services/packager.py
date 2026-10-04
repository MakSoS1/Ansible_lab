import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from config import DANCES_DIR


def _portable_path(path: Optional[str], dance_dir: str) -> str:
    if not path:
        return ""
    absolute = os.path.abspath(path)
    root = os.path.abspath(dance_dir)
    try:
        if os.path.commonpath([absolute, root]) == root:
            return os.path.relpath(absolute, root).replace(os.sep, "/")
    except ValueError:
        pass
    return path


def _resolved_path(path: Optional[str], dance_dir: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(dance_dir, path.replace("/", os.sep)))


def save_dance_package(dance_id: str, data: Dict[str, Any]) -> str:
    dance_dir = os.path.join(DANCES_DIR, dance_id)
    os.makedirs(dance_dir, exist_ok=True)
    os.makedirs(os.path.join(dance_dir, "assets"), exist_ok=True)
    os.makedirs(os.path.join(dance_dir, "audio"), exist_ok=True)

    manifest = {
        "dance_id": dance_id,
        "title": data.get("title", dance_id),
        "version": data.get("version", 3),
        "duration_ms": data.get("duration_ms", 0),
        "skeleton_format": "blazepose_33",
        "preview_mode": data.get("preview_mode", "stylized_game_video"),
        "difficulty": data.get("difficulty", "medium"),
        "mirror_mode": data.get("mirror_mode", True),
        "created_at": data.get("created_at", ""),
        "num_frames": data.get("num_frames", 0),
        "num_events": data.get("num_events", 0),
        "source_type": data.get("source_type", "file"),
        "source_url": data.get("source_url", ""),
        "pose_coverage": data.get("pose_coverage", 0.0),
    }
    with open(os.path.join(dance_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    for key, filename in (
        ("reference_pose", "reference_pose.json"),
        ("timing", "timing.json"),
        ("events", "events.json"),
        ("weights", "weights.json"),
        ("theme", "theme.json"),
        ("render", "render.json"),
    ):
        if key in data:
            with open(os.path.join(dance_dir, filename), "w", encoding="utf-8") as f:
                json.dump(data[key], f, indent=2, ensure_ascii=False)

    preview = dict(data.get("preview", {
        "video_path": data.get("video_path", ""),
        "audio_path": data.get("audio_path", ""),
    }))
    for key in ("video_path", "audio_path", "poster_path", "source_preview_path"):
        preview[key] = _portable_path(preview.get(key), dance_dir)
    with open(os.path.join(dance_dir, "preview.json"), "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2)

    return dance_dir


def load_dance_package(dance_id: str, load_pose: bool = True) -> Optional[Dict[str, Any]]:
    dance_dir = os.path.join(DANCES_DIR, dance_id)
    if not os.path.isdir(dance_dir):
        return None

    result: Dict[str, Any] = {}
    load_files = ["manifest.json", "timing.json", "events.json", "weights.json", "theme.json", "render.json", "preview.json"]
    if load_pose:
        load_files.insert(1, "reference_pose.json")

    for fname in load_files:
        fpath = os.path.join(dance_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                key = fname.replace(".json", "")
                if key == "manifest":
                    result.update(json.load(f))
                else:
                    result[key] = json.load(f)

    preview = result.get("preview")
    if isinstance(preview, dict):
        for key in ("video_path", "audio_path", "poster_path", "source_preview_path"):
            preview[key] = _resolved_path(preview.get(key), dance_dir)

    result["dance_dir"] = dance_dir
    return result


def export_dance_pack(dance_id: str, output_path: str) -> str:
    dance_dir = os.path.join(DANCES_DIR, dance_id)
    if not os.path.isdir(dance_dir):
        raise ValueError(f"Dance not found: {dance_id}")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(dance_dir):
            for file in files:
                fpath = os.path.join(root, file)
                arcname = os.path.relpath(fpath, dance_dir)
                zf.write(fpath, arcname)
    return output_path


def _safe_extract(zf: zipfile.ZipFile, destination: str) -> None:
    root = Path(destination).resolve()
    for info in zf.infolist():
        candidate = (root / info.filename).resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError(f"Unsafe path in dance pack: {info.filename}")
    zf.extractall(destination)


def import_dance_pack(pack_path: str, dance_id: Optional[str] = None) -> str:
    with zipfile.ZipFile(pack_path, "r") as zf:
        manifest_data = None
        for name in zf.namelist():
            if name == "manifest.json" or name.endswith("/manifest.json"):
                manifest_data = json.loads(zf.read(name))
                break
        if not manifest_data:
            raise ValueError("No manifest.json in pack")

        did = dance_id or str(manifest_data.get("dance_id", "imported"))
        if not did or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in did):
            raise ValueError("Invalid dance_id in pack")
        dance_dir = os.path.join(DANCES_DIR, did)
        os.makedirs(dance_dir, exist_ok=True)
        _safe_extract(zf, dance_dir)
    return did


def delete_dance_package(dance_id: str):
    dance_dir = os.path.join(DANCES_DIR, dance_id)
    if os.path.isdir(dance_dir):
        shutil.rmtree(dance_dir)
