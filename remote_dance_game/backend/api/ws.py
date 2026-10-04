import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import database as db
from config import DANCES_DIR
from services.calibration import run_calibration
from services.packager import load_dance_package
from services.scoring_engine import ScoringEngine

router = APIRouter()
active_sessions: Dict[str, Dict[str, Any]] = {}


def _load_ref_frames(dance_id: str) -> list:
    pose_path = os.path.join(DANCES_DIR, dance_id, "reference_pose.json")
    if not os.path.exists(pose_path):
        return []
    with open(pose_path, "r", encoding="utf-8") as f:
        return json.load(f).get("frames", [])


def create_game_session(dance_id: str) -> Dict[str, Any]:
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    dance_pack = load_dance_package(dance_id, load_pose=False)
    if not dance_pack:
        raise ValueError(f"Dance package not found: {dance_id}")
    dance_pack["reference_pose"] = {"frames": []}
    engine = ScoringEngine(dance_pack)
    session = {
        "session_id": session_id,
        "dance_id": dance_id,
        "engine": engine,
        "desktop_ws": None,
        "phone_ws": None,
        "phone_connected": False,
        "calibrated": False,
        "state": "created",
        "latest_score": None,
        "pose_loaded": False,
        "last_media_ms": 0,
        "result_saved": False,
    }
    active_sessions[session_id] = session
    return session


def _ensure_pose_loaded(session: Dict[str, Any]):
    if session["pose_loaded"]:
        return
    ref_frames = _load_ref_frames(session["dance_id"])
    session["engine"].ref_frames = ref_frames
    session["pose_loaded"] = True


async def _broadcast_score(session: Dict[str, Any], score_event):
    event_dict = score_event.dict()
    session["latest_score"] = event_dict
    desktop_ws = session.get("desktop_ws")
    if desktop_ws:
        try:
            await desktop_ws.send_json(event_dict)
        except Exception:
            pass
    phone_ws = session.get("phone_ws")
    if phone_ws:
        try:
            await phone_ws.send_json({
                "type": "score_feedback",
                "grade": score_event.grade.value,
                "score": score_event.score,
                "total_score": score_event.total_score,
                "combo": score_event.combo,
                "similarity": score_event.similarity,
                "hold_state": score_event.hold_state,
                "tracking_lost": score_event.tracking_lost,
                "is_move_grade": score_event.is_move_grade,
                "move_index": score_event.move_index,
                "move_count": score_event.move_count,
            })
        except Exception:
            pass


async def _finish_session(session: Dict[str, Any], reason: str = "completed"):
    engine: ScoringEngine = session["engine"]
    if session.get("result_saved"):
        return engine.get_final_results()
    engine.stop()
    session["state"] = "stopped"
    await db.update_session(session["session_id"], status="stopped")
    results = engine.get_final_results()
    await _save_results(session["session_id"], session["dance_id"], results)
    session["result_saved"] = True
    desktop_ws = session.get("desktop_ws")
    if desktop_ws:
        try:
            await desktop_ws.send_json({"type": "game_over", "reason": reason, "results": results})
        except Exception:
            pass
    phone_ws = session.get("phone_ws")
    if phone_ws:
        try:
            await phone_ws.send_json({"type": "game_over", "reason": reason, "results": results})
        except Exception:
            pass
    return results


@router.websocket("/ws/game/{session_id}")
async def game_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = active_sessions.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return
    session["desktop_ws"] = websocket

    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            action = msg.get("action")

            if action == "start":
                await asyncio.to_thread(_ensure_pose_loaded, session)
                engine: ScoringEngine = session["engine"]
                engine.start()
                session["state"] = "playing"
                session["last_media_ms"] = int(msg.get("media_time_ms", 0) or 0)
                session["result_saved"] = False
                await db.update_session(session_id, status="playing")
                phone_ws = session.get("phone_ws")
                if phone_ws:
                    try:
                        await phone_ws.send_json({"type": "start"})
                    except Exception:
                        pass
                await websocket.send_json({"type": "started", "server_timestamp_ms": int(time.time() * 1000)})

            elif action == "media_clock":
                engine: ScoringEngine = session["engine"]
                if not engine.is_running or engine.is_paused:
                    continue
                media_ms = max(0, int(msg.get("media_time_ms", 0) or 0))
                session["last_media_ms"] = media_ms
                if engine.duration_ms and media_ms >= max(0, engine.duration_ms - 20):
                    await _finish_session(session, reason="media_ended")
                    continue
                score_event = engine.score_tick(media_ms, now_server_ms=int(time.time() * 1000))
                if score_event:
                    await _broadcast_score(session, score_event)

            elif action == "pause":
                session["engine"].pause()
                session["state"] = "paused"
                await db.update_session(session_id, status="paused")
                if session.get("phone_ws"):
                    try:
                        await session["phone_ws"].send_json({"type": "pause"})
                    except Exception:
                        pass

            elif action == "resume":
                session["engine"].resume()
                session["state"] = "playing"
                await db.update_session(session_id, status="playing")
                if session.get("phone_ws"):
                    try:
                        await session["phone_ws"].send_json({"type": "resume"})
                    except Exception:
                        pass

            elif action == "stop":
                await _finish_session(session, reason="stopped_by_user")

            elif action == "calibrate":
                session["state"] = "calibrating"
                session["calibrated"] = False
                await db.update_session(session_id, status="calibrating")

    except WebSocketDisconnect:
        session["desktop_ws"] = None
    except Exception as exc:
        print(f"Desktop WS error: {exc}")
        session["desktop_ws"] = None


@router.websocket("/ws/phone/{session_id}")
async def phone_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = active_sessions.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    session["phone_ws"] = websocket
    session["phone_connected"] = True
    await db.update_session(session_id, status="phone_connected")
    if session.get("desktop_ws"):
        try:
            await session["desktop_ws"].send_json({"type": "phone_connected"})
        except Exception:
            pass

    calibration_frames = []
    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            msg_type = msg.get("type")

            if msg_type == "pose_frame":
                engine: ScoringEngine = session.get("engine")
                if engine and engine.is_running:
                    received_at_ms = int(time.time() * 1000)
                    one_way_ms = max(0, min(220, int(msg.get("estimated_latency_ms", 0) or 0)))
                    # Desktop media clock is refreshed ~25 Hz. Subtract the
                    # measured one-way network delay to estimate where the song
                    # was when the camera actually captured this pose.
                    media_capture_ms = max(
                        0,
                        int(session.get("last_media_ms", 0) or 0) - one_way_ms,
                    )
                    engine.add_pose_frame({
                        "timestamp_ms": msg.get("timestamp_ms", received_at_ms),
                        "received_at_ms": received_at_ms,
                        "media_capture_ms": media_capture_ms,
                        "network_latency_ms": one_way_ms,
                        "landmarks": msg.get("landmarks", []),
                        "world_landmarks": msg.get("world_landmarks", []),
                        "tracking_score": msg.get("tracking_score", 0.0),
                    })

            elif msg_type == "calibration_frame":
                calibration_frames.append(msg.get("landmarks", []))
                if len(calibration_frames) >= 10:
                    result = run_calibration(calibration_frames)
                    calibration_frames = []
                    result_dict = result.dict()
                    session["calibrated"] = bool(result_dict.get("ready", False))
                    session["state"] = "calibrated" if session["calibrated"] else "calibrating"
                    await db.update_session(session_id, status=session["state"])
                    await websocket.send_json(result_dict)
                    if session.get("desktop_ws"):
                        try:
                            await session["desktop_ws"].send_json(result_dict)
                        except Exception:
                            pass

            elif msg_type == "control" and msg.get("action") == "ready":
                session["calibrated"] = True
                if session.get("state") != "playing":
                    session["state"] = "calibrated"
                    await db.update_session(session_id, status="calibrated")

            elif msg_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "client_timestamp_ms": msg.get("timestamp_ms"),
                    "server_timestamp_ms": int(time.time() * 1000),
                })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"Phone WS error: {exc}")
    finally:
        session["phone_ws"] = None
        session["phone_connected"] = False
        session["calibrated"] = False
        if session.get("state") != "stopped":
            session["state"] = "phone_disconnected"
            await db.update_session(session_id, status="phone_disconnected")
        if session.get("desktop_ws"):
            try:
                await session["desktop_ws"].send_json({"type": "phone_disconnected"})
            except Exception:
                pass


async def _save_results(session_id: str, dance_id: str, results: Dict[str, Any]):
    await db.insert_result({
        "session_id": session_id,
        "dance_id": dance_id,
        "total_score": results.get("total_score", 0),
        "max_combo": results.get("max_combo", 0),
        "grade_counts": results.get("grade_counts", {}),
        "hold_results": results.get("hold_results", []),
        "accuracy_arms": results.get("accuracy_arms", 0.0),
        "accuracy_legs": results.get("accuracy_legs", 0.0),
        "accuracy_torso": results.get("accuracy_torso", 0.0),
        "timeline_scores": results.get("timeline_scores", []),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
