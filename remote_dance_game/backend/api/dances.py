import os
import re
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

import database as db
from config import EXPORTS_DIR, INPUT_DIR
from models.schemas import DanceCreateRequest, DanceDetailResponse, DanceListItem, JobStatusResponse
from services.packager import delete_dance_package, export_dance_pack, import_dance_pack, load_dance_package

router = APIRouter(prefix="/api", tags=["dances"])


def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "video.mp4")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned or "video.mp4"


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
    return [
        DanceListItem(
            dance_id=d["dance_id"],
            title=d["title"],
            duration_ms=d.get("duration_ms", 0),
            difficulty=d.get("difficulty", "medium"),
            created_at=d["created_at"],
            has_video=bool((d.get("video_path") or "") and os.path.exists(d.get("video_path") or "")),
        )
        for d in dances
    ]


@router.get("/dances/{dance_id}", response_model=DanceDetailResponse)
async def get_dance(dance_id: str):
    d = await db.get_dance(dance_id)
    if not d:
        raise HTTPException(status_code=404, detail="Dance not found")
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
    )


@router.get("/dances/{dance_id}/playback")
async def get_playback(dance_id: str):
    pack = load_dance_package(dance_id, load_pose=False)
    if not pack:
        raise HTTPException(status_code=404, detail="Dance not found")
    timing = pack.get("timing", {})
    events = pack.get("events", {}).get("events", [])
    return {
        "dance_id": dance_id,
        "duration_ms": int(pack.get("duration_ms", 0) or 0),
        "mirror_mode": bool(pack.get("mirror_mode", True)),
        "tempo": float(timing.get("tempo", 120) or 120),
        "beat_ms": timing.get("beat_ms", []),
        "strong_beat_ms": timing.get("strong_beat_ms", []),
        "events": events[:200],
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
