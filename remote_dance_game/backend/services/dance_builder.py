import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import database as db
from config import DANCES_DIR
from services.choreography import process_choreography
from services.events import detect_hold_events
from services.packager import save_dance_package
from services.pose_extractor import extract_poses_from_video
from services.source_resolver import resolve_source
from services.timing import extract_timing
from services.video_processor import create_preview_video, extract_audio, normalize_video, probe_video
from services.weights import generate_weights


async def build_dance(
    request_data: Dict[str, Any],
    progress_cb: Optional[Callable] = None,
    job_id: Optional[str] = None,
    dance_id: Optional[str] = None,
) -> Dict[str, Any]:
    job_id = job_id or f"job_{uuid.uuid4().hex[:8]}"
    dance_id = dance_id or f"dance_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()

    existing_job = await db.get_job(job_id)
    if not existing_job:
        await db.insert_job({
            "job_id": job_id,
            "dance_id": dance_id,
            "status": "queued",
            "progress": 0,
            "stage": "starting",
            "created_at": now,
            "updated_at": now,
        })

    async def update_job(progress: int, stage: str, status: Optional[str] = None, error: Optional[str] = None):
        payload: Dict[str, Any] = {
            "progress": progress,
            "stage": stage,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if status is not None:
            payload["status"] = status
        if error is not None:
            payload["error"] = error
        await db.update_job(job_id, **payload)

    def progress(pct: int, stage: str):
        if progress_cb:
            progress_cb(pct, stage)

    try:
        clip_start = max(0.0, float(request_data.get("clip_start_sec", 0.0) or 0.0))
        clip_end = request_data.get("clip_end_sec")
        clip_end = float(clip_end) if clip_end is not None else None
        if clip_end is not None and clip_end <= clip_start:
            raise ValueError("Clip end must be greater than clip start")

        mirror_mode = bool(request_data.get("mirror_mode", True))
        difficulty = str(request_data.get("difficulty", "medium"))
        source_type = str(request_data.get("source_type", "file"))
        source_url = request_data.get("source_url") or request_data.get("youtube_url")

        dance_dir = os.path.join(DANCES_DIR, dance_id)
        assets_dir = os.path.join(dance_dir, "assets")
        audio_dir = os.path.join(dance_dir, "audio")
        source_dir = os.path.join(dance_dir, "source")
        os.makedirs(assets_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(source_dir, exist_ok=True)

        progress(2, "resolving_source")
        await update_job(2, "resolving_source", status="running")

        def source_progress(download_pct: int, stage: str):
            mapped = 2 + int(max(0, min(100, download_pct)) * 0.08)
            progress(mapped, stage)

        file_path, detected_title = await asyncio.to_thread(
            resolve_source,
            source_type,
            request_data.get("file_path"),
            source_url,
            source_dir,
            source_progress,
        )
        title = str(request_data.get("title") or detected_title or os.path.splitext(os.path.basename(file_path))[0])

        progress(10, "validating")
        await update_job(10, "validating")
        metadata = await asyncio.to_thread(probe_video, file_path)
        if not metadata.get("streams"):
            raise ValueError("No media streams found in source video")

        progress(15, "normalizing_video")
        await update_job(15, "normalizing_video")
        proxy_path = os.path.join(dance_dir, "proxy.mp4")
        await asyncio.to_thread(normalize_video, file_path, proxy_path, clip_start, clip_end)

        progress(23, "extracting_audio")
        await update_job(23, "extracting_audio")
        audio_path = os.path.join(audio_dir, "track.mp3")
        try:
            await asyncio.to_thread(extract_audio, file_path, audio_path, clip_start, clip_end)
        except Exception:
            audio_path = None

        progress(28, "creating_preview")
        await update_job(28, "creating_preview")
        preview_path = os.path.join(assets_dir, "preview.mp4")
        try:
            await asyncio.to_thread(create_preview_video, file_path, preview_path, clip_start, clip_end)
        except Exception:
            preview_path = proxy_path

        progress(32, "pose_extraction")
        await update_job(32, "pose_extraction")

        def pose_progress(pct, stage):
            adjusted = 32 + int(pct * 0.36)
            progress(adjusted, stage)

        pose_data = await asyncio.to_thread(
            extract_poses_from_video,
            proxy_path,
            progress_cb=pose_progress,
            mirror_mode=mirror_mode,
        )

        valid_frames = sum(
            1 for frame in pose_data.get("frames", [])
            if frame.get("landmarks") and sum(float(x.get("v", 0.0)) for x in frame["landmarks"][11:29]) / 18.0 >= 0.35
        )
        total_frames = max(1, int(pose_data.get("total_frames", 0) or 0))
        pose_coverage = valid_frames / total_frames
        if pose_coverage < 0.35:
            raise ValueError(
                f"The dancer is not visible reliably enough (pose coverage {pose_coverage:.0%}). "
                "Use a clip where the main dancer's full body is visible for most of the video."
            )

        progress(70, "processing_choreography")
        await update_job(70, "processing_choreography")
        pose_data = await asyncio.to_thread(process_choreography, pose_data)

        progress(80, "extracting_timing")
        await update_job(80, "extracting_timing")
        timing_data = {
            "tempo": 120,
            "duration_ms": pose_data["duration_ms"],
            "beat_ms": [],
            "onset_ms": [],
            "strong_beat_ms": [],
        }
        if audio_path and os.path.exists(audio_path):
            try:
                timing_data = await asyncio.to_thread(extract_timing, audio_path)
            except Exception:
                pass

        progress(86, "generating_weights")
        await update_job(86, "generating_weights")
        weights_data = await asyncio.to_thread(generate_weights, pose_data["frames"])

        progress(91, "detecting_events")
        await update_job(91, "detecting_events")
        events_list = await asyncio.to_thread(detect_hold_events, pose_data["frames"])

        progress(96, "packaging")
        await update_job(96, "packaging")
        pack_data = {
            "title": title,
            "version": 2,
            "duration_ms": pose_data["duration_ms"],
            "preview_mode": "local_video",
            "difficulty": difficulty,
            "mirror_mode": mirror_mode,
            "created_at": now,
            "source_type": source_type,
            "source_url": source_url or "",
            "pose_coverage": round(pose_coverage, 4),
            "num_frames": pose_data["total_frames"],
            "num_events": len(events_list),
            "reference_pose": {"fps": pose_data["fps"], "frames": pose_data["frames"]},
            "timing": timing_data,
            "events": {"events": events_list},
            "weights": weights_data,
            "preview": {"video_path": preview_path, "audio_path": audio_path or ""},
            "video_path": preview_path,
            "audio_path": audio_path,
        }

        dance_dir = await asyncio.to_thread(save_dance_package, dance_id, pack_data)
        await db.insert_dance({
            "dance_id": dance_id,
            "title": title,
            "version": 2,
            "duration_ms": pose_data["duration_ms"],
            "difficulty": difficulty,
            "mirror_mode": mirror_mode,
            "created_at": now,
            "num_frames": pose_data["total_frames"],
            "num_events": len(events_list),
            "video_path": preview_path,
            "audio_path": audio_path,
            "dance_dir": dance_dir,
        })
        await db.update_job(
            job_id,
            status="completed",
            progress=100,
            stage="done",
            dance_id=dance_id,
            updated_at=datetime.now(timezone.utc).isoformat(),
            error=None,
        )
        progress(100, "done")
        return {"job_id": job_id, "dance_id": dance_id, "status": "completed"}

    except Exception as exc:
        await db.update_job(
            job_id,
            status="failed",
            error=str(exc),
            stage="failed",
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        raise
