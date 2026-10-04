import os
import re
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

import database as db
from config import EXPORTS_DIR, INPUT_DIR
from models.schemas import DanceCreateRequest, DanceDetailResponse, DanceListItem, JobStatusResponse
from services.packager import delete_dance_package, export_dance_pack, import_dance_pack, load_dance_package
from services.move_events import build_move_markers, build_pictogram_markers

router = APIRouter(prefix="/api", tags=["dances"])


def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "video.mp4")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned or "video.mp4"


def _pack_extras(dance_id: str):
    pack = load_dance_package(dance_id, load_pose=False) or {}
    preview = pack.get("preview", {}) if isinstance(pack.get("preview"), dict) else {}
    poster_path = preview.get("poster_path") or ""
    return pack, bool(poster_path and os.path.exists(poster_path))


@router.post("/uploads/video")
async def upload_video(file: UploadFile = File(...)):
    allowed = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
    filename = _safe_filename(file.filename or "video.mp4")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported video extension: {ext or 'none'}")
    os.makedirs(INPUT_DIR, exist_ok=True)
    output_path = os.path.join(INPUT_DIR, f"{uuid.uuid4().hex[:10]}_{filename}")
    size = 0
    max_bytes = 2 * 1024 * 1024 * 1024
    try:
        with open(output_path, "wb") as target:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail="Video is larger than 2 GiB")
                target.write(chunk)
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        await file.close()
    return {"file_path": output_path, "filename": filename, "size_bytes": size}


@router.post("/dances")
async def create_dance(req: DanceCreateRequest):
    import asyncio

    source_type = req.source_type.value
    source_url = req.source_url or req.youtube_url
    if source_type == "file":
        if not req.file_path or not os.path.isfile(req.file_path):
            raise HTTPException(status_code=400, detail=f"File not found: {req.file_path or ''}")
    elif not source_url:
        raise HTTPException(status_code=400, detail="Video URL is required")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    dance_id = f"dance_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    await db.insert_job({
        "job_id": job_id,
        "dance_id": dance_id,
        "status": "queued",
        "progress": 0,
        "stage": "starting",
        "created_at": now,
        "updated_at": now,
    })
    asyncio.create_task(_run_build(job_id, dance_id, req))
    return {"job_id": job_id, "dance_id": dance_id, "status": "queued"}


async def _run_build(job_id: str, dance_id: str, req: DanceCreateRequest):
    from services.dance_builder import build_dance
    try:
        await build_dance(
            {
                "source_type": req.source_type.value,
                "file_path": req.file_path,
                "source_url": req.source_url or req.youtube_url,
                "clip_start_sec": req.clip_start_sec,
                "clip_end_sec": req.clip_end_sec,
                "mirror_mode": req.mirror_mode,
                "title": req.title,
                "difficulty": req.difficulty,
            },
            job_id=job_id,
            dance_id=dance_id,
        )
    except Exception as exc:
        await db.update_job(job_id, status="failed", error=str(exc), updated_at=datetime.now(timezone.utc).isoformat())


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job.get("progress", 0),
        stage=job.get("stage", ""),
        error=job.get("error"),
        dance_id=job.get("dance_id"),
    )


@router.get("/dances", response_model=List[DanceListItem])
async def list_dances():
    dances = await db.list_dances()
    result = []
    for d in dances:
        pack, has_poster = _pack_extras(d["dance_id"])
        result.append(DanceListItem(
            dance_id=d["dance_id"],
            title=d["title"],
            duration_ms=d.get("duration_ms", 0),
            difficulty=d.get("difficulty", "medium"),
            created_at=d["created_at"],
            has_video=bool((d.get("video_path") or "") and os.path.exists(d.get("video_path") or "")),
            has_poster=has_poster,
            preview_mode=d.get("preview_mode", pack.get("preview_mode", "local_video")),
            theme=pack.get("theme", {}) if isinstance(pack.get("theme"), dict) else {},
            coach_count=int(pack.get("coach_count", ((pack.get("coaches") or {}).get("coach_count", 1)) or 1)),
        ))
    return result


@router.get("/dances/{dance_id}", response_model=DanceDetailResponse)
async def get_dance(dance_id: str):
    d = await db.get_dance(dance_id)
    if not d:
        raise HTTPException(status_code=404, detail="Dance not found")
    pack, has_poster = _pack_extras(dance_id)
    return DanceDetailResponse(
        dance_id=d["dance_id"],
        title=d["title"],
        version=d.get("version", 1),
        duration_ms=d.get("duration_ms", 0),
        skeleton_format=d.get("skeleton_format", "blazepose_33"),
        preview_mode=d.get("preview_mode", "local_video"),
        difficulty=d.get("difficulty", "medium"),
        mirror_mode=d.get("mirror_mode", True),
        created_at=d["created_at"],
        num_frames=d.get("num_frames", 0),
        num_events=d.get("num_events", 0),
        video_path=d.get("video_path"),
        audio_path=d.get("audio_path"),
        has_poster=has_poster,
        theme=pack.get("theme", {}) if isinstance(pack.get("theme"), dict) else {},
        coach_count=int(pack.get("coach_count", ((pack.get("coaches") or {}).get("coach_count", 1)) or 1)),
    )


@router.get("/dances/{dance_id}/poster")
async def get_dance_poster(dance_id: str):
    pack = load_dance_package(dance_id, load_pose=False)
    if not pack:
        raise HTTPException(status_code=404, detail="Dance not found")
    poster_path = (pack.get("preview") or {}).get("poster_path", "")
    if not poster_path or not os.path.exists(poster_path):
        raise HTTPException(status_code=404, detail="Poster not found")
    return FileResponse(poster_path, media_type="image/jpeg")


@router.get("/dances/{dance_id}/playback")
async def get_playback(dance_id: str):
    """Gameplay metadata plus sparse Just-Dance-style pictogram cues."""
    pack = load_dance_package(dance_id, load_pose=True)
    if not pack:
        raise HTTPException(status_code=404, detail="Dance not found")

    timing = pack.get("timing", {})
    events = pack.get("events", {}).get("events", [])
    reference = pack.get("reference_pose") or {}
    primary_frames = reference.get("frames") or []
    tracks = reference.get("tracks") or []
    if not tracks:
        tracks = [{
            "coach_index": 0,
            "frames": primary_frames,
            "weights": pack.get("weights", []),
            "events": events,
            "coverage": float(pack.get("pose_coverage", 1.0) or 1.0),
        }]

    preview_indices = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32}

    def compact_pose(source):
        compact = []
        for idx in range(33):
            if idx in preview_indices and idx < len(source):
                lm = source[idx] or {}
                compact.append({
                    "x": round(float(lm.get("x", 0.0)), 4),
                    "y": round(float(lm.get("y", 0.0)), 4),
                    "v": round(float(lm.get("v", 0.0)), 3),
                })
            else:
                compact.append({"x": 0.0, "y": 0.0, "v": 0.0})
        return compact

    def motion_hints(frames, idx):
        if not frames or idx < 0 or idx >= len(frames):
            return []
        target_t = int(frames[idx].get("t_ms", 0))
        prev_idx = idx
        while prev_idx > 0 and target_t - int(frames[prev_idx].get("t_ms", 0)) < 320:
            prev_idx -= 1
        current = frames[idx].get("landmarks", []) or []
        previous = frames[prev_idx].get("landmarks", []) or []
        hints = []
        for joint in (15, 16, 27, 28):
            if joint >= len(current) or joint >= len(previous):
                continue
            a = previous[joint] or {}
            b = current[joint] or {}
            if min(float(a.get("v", 0.0)), float(b.get("v", 0.0))) < 0.22:
                continue
            dx = float(b.get("x", 0.0)) - float(a.get("x", 0.0))
            dy = float(b.get("y", 0.0)) - float(a.get("y", 0.0))
            mag = (dx * dx + dy * dy) ** 0.5
            if mag < 0.035:
                continue
            hints.append({
                "joint": joint,
                "dx": round(dx, 4),
                "dy": round(dy, 4),
                "magnitude": round(min(mag, 0.45), 4),
            })
        return hints

    coach_cues = []
    for coach_index, track in enumerate(tracks[:4]):
        frames = track.get("frames") or []
        cues = []
        for marker in build_pictogram_markers(frames, timing)[:240]:
            idx = int(marker.get("frame_index", 0))
            if idx < 0 or idx >= len(frames):
                continue
            cues.append({
                "t_ms": int(marker.get("t_ms", 0)),
                "cue_index": int(marker.get("cue_index", len(cues))),
                "motion": float(marker.get("motion", 0.0)),
                "landmarks": compact_pose(frames[idx].get("landmarks", []) or []),
                "motion_hints": motion_hints(frames, idx),
            })
        coach_cues.append({
            "coach_index": coach_index,
            "cues": cues,
        })

    # Dense move markers are still exposed for debugging/analytics, while the TV
    # HUD consumes coach_cues. This intentionally decouples grading cadence from
    # visual-instruction cadence.
    move_previews = []
    for marker in build_move_markers(primary_frames, timing)[:720]:
        idx = int(marker.get("frame_index", 0))
        if idx < 0 or idx >= len(primary_frames):
            continue
        move_previews.append({
            "t_ms": int(marker.get("t_ms", 0)),
            "move_index": int(marker.get("move_index", len(move_previews))),
            "motion": float(marker.get("motion", 0.0)),
            "landmarks": compact_pose(primary_frames[idx].get("landmarks", []) or []),
        })

    coaches = pack.get("coaches") if isinstance(pack.get("coaches"), dict) else {}
    return {
        "dance_id": dance_id,
        "duration_ms": int(pack.get("duration_ms", 0) or 0),
        "mirror_mode": bool(pack.get("mirror_mode", True)),
        "tempo": float(timing.get("tempo", 120) or 120),
        "beat_ms": timing.get("beat_ms", []),
        "strong_beat_ms": timing.get("strong_beat_ms", []),
        "events": events[:200],
        "theme": pack.get("theme", {}),
        "move_previews": move_previews,
        "move_count": len(move_previews),
        "coach_count": max(1, len(coach_cues)),
        "coach_cues": coach_cues,
        "coaches": coaches.get("items", []),
    }


@router.get("/dances/{dance_id}/coaches")
async def get_dance_coaches(dance_id: str):
    pack = load_dance_package(dance_id, load_pose=False)
    if not pack:
        raise HTTPException(status_code=404, detail="Dance not found")
    coaches = pack.get("coaches") if isinstance(pack.get("coaches"), dict) else {}
    items = coaches.get("items", [])
    return {
        "dance_id": dance_id,
        "coach_count": int(pack.get("coach_count", coaches.get("coach_count", max(1, len(items)))) or 1),
        "coaches": items,
    }


@router.get("/dances/{dance_id}/coaches/{coach_index}/preview")
async def get_dance_coach_preview(dance_id: str, coach_index: int):
    pack = load_dance_package(dance_id, load_pose=False)
    if not pack:
        raise HTTPException(status_code=404, detail="Dance not found")
    coaches = pack.get("coaches") if isinstance(pack.get("coaches"), dict) else {}
    items = coaches.get("items", [])
    item = next((x for x in items if int(x.get("coach_index", -1)) == coach_index), None)
    if not item:
        raise HTTPException(status_code=404, detail="Coach preview not found")
    rel = str(item.get("preview_path", ""))
    path = os.path.abspath(os.path.join(pack["dance_dir"], rel.replace("/", os.sep)))
    root = os.path.abspath(pack["dance_dir"])
    if not path.startswith(root) or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Coach preview not found")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/dances/{dance_id}/pose-timeline")
async def get_pose_timeline(dance_id: str, fps: int = 20):
    """Downsampled world-space choreography for Unity Humanoid retargeting.

    Scoring still uses the full reference timeline server-side. Unity only needs
    a presentation stream, so limiting this endpoint to 5..30 FPS keeps payloads
    manageable even for multi-minute songs.
    """
    pack = load_dance_package(dance_id, load_pose=True)
    if not pack:
        raise HTTPException(status_code=404, detail="Dance not found")

    fps = max(5, min(30, int(fps)))
    reference = pack.get("reference_pose") or {}
    source_frames = reference.get("frames") or []
    target_step_ms = max(1, int(round(1000.0 / fps)))
    frames = []
    last_t = -target_step_ms

    for frame in source_frames:
        t_ms = int(frame.get("t_ms", 0) or 0)
        if t_ms - last_t < target_step_ms:
            continue
        world = frame.get("world_landmarks") or []
        normalized = frame.get("landmarks") or []
        source = world if len(world) >= 29 else normalized
        if len(source) < 29:
            continue
        landmarks = []
        for idx in range(33):
            lm = source[idx] if idx < len(source) else {}
            landmarks.append({
                "x": round(float(lm.get("x", 0.0)), 5),
                "y": round(float(lm.get("y", 0.0)), 5),
                "z": round(float(lm.get("z", 0.0)), 5),
                "v": round(float(lm.get("v", 0.0)), 4),
            })
        frames.append({"t_ms": t_ms, "landmarks": landmarks})
        last_t = t_ms

    return {
        "dance_id": dance_id,
        "fps": fps,
        "duration_ms": int(pack.get("duration_ms", 0) or 0),
        "space": "world_or_normalized",
        "frames": frames,
    }


@router.delete("/dances/{dance_id}")
async def delete_dance(dance_id: str):
    d = await db.get_dance(dance_id)
    if not d:
        raise HTTPException(status_code=404, detail="Dance not found")
    delete_dance_package(dance_id)
    await db.delete_dance(dance_id)
    return {"status": "deleted"}


@router.get("/dances/{dance_id}/download")
async def export_dance(dance_id: str):
    d = await db.get_dance(dance_id)
    if not d:
        raise HTTPException(status_code=404, detail="Dance not found")
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    out_path = os.path.join(EXPORTS_DIR, f"{dance_id}.dancepack")
    export_dance_pack(dance_id, out_path)
    return {"download_path": out_path}


@router.post("/dances/import")
async def import_dance(file: UploadFile = File(...)):
    tmp_path = os.path.join(EXPORTS_DIR, f"import_{uuid.uuid4().hex[:6]}.dancepack")
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    try:
        dance_id = import_dance_pack(tmp_path)
        pack = load_dance_package(dance_id)
        if pack:
            now = datetime.now(timezone.utc).isoformat()
            await db.insert_dance({
                "dance_id": dance_id,
                "title": pack.get("title", dance_id),
                "version": pack.get("version", 1),
                "duration_ms": pack.get("duration_ms", 0),
                "difficulty": pack.get("difficulty", "medium"),
                "mirror_mode": pack.get("mirror_mode", True),
                "preview_mode": pack.get("preview_mode", "local_video"),
                "created_at": pack.get("created_at", now),
                "num_frames": pack.get("num_frames", 0),
                "num_events": pack.get("num_events", 0),
                "video_path": pack.get("preview", {}).get("video_path"),
                "audio_path": pack.get("preview", {}).get("audio_path"),
                "dance_dir": pack.get("dance_dir", ""),
            })
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return {"dance_id": dance_id, "status": "imported"}
