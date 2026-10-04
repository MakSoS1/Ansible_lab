import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import database as db
from config import DANCES_DIR
from services.calibration import run_calibration
from services.packager import load_dance_package
from services.scoring_engine import ScoringEngine

router = APIRouter()
active_sessions: Dict[str, Dict[str, Any]] = {}


def _load_reference_pose(dance_id: str) -> Dict[str, Any]:
    pose_path = os.path.join(DANCES_DIR, dance_id, "reference_pose.json")
    if not os.path.exists(pose_path):
        return {"frames": [], "tracks": []}
    with open(pose_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_game_session(dance_id: str) -> Dict[str, Any]:
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    dance_pack = load_dance_package(dance_id, load_pose=False)
    if not dance_pack:
        raise ValueError(f"Dance package not found: {dance_id}")

    coaches = dance_pack.get("coaches") if isinstance(dance_pack.get("coaches"), dict) else {}
    coach_count = int(dance_pack.get("coach_count", coaches.get("coach_count", 1)) or 1)

    session = {
        "session_id": session_id,
        "dance_id": dance_id,
        "dance_pack": dance_pack,
        "coach_count": max(1, coach_count),
        "coach_tracks": [],
        "reference_loaded": False,
        "player_engines": {},
        "players": {},
        "assignments": {},
        "next_player_slot": 0,
        "desktop_ws": None,
        "phone_ws": None,
        "phone_connected": False,
        "calibrated": False,
        "state": "created",
        "latest_score": None,
        "last_media_ms": 0,
        "result_saved": False,
        "calibration_frames": {},
        "last_pose_broadcast_ms": {},
    }
    active_sessions[session_id] = session
    return session


def _ensure_reference_loaded(session: Dict[str, Any]) -> None:
    if session.get("reference_loaded"):
        return

    reference = _load_reference_pose(session["dance_id"])
    tracks = reference.get("tracks") or []
    if not tracks:
        tracks = [{
            "coach_index": 0,
            "source_track_id": 0,
            "frames": reference.get("frames", []),
            "weights": session["dance_pack"].get("weights", []),
            "events": (session["dance_pack"].get("events") or {}).get("events", []),
        }]

    normalized = []
    for index, track in enumerate(tracks):
        normalized.append({
            "coach_index": index,
            "source_track_id": int(track.get("source_track_id", track.get("track_id", index))),
            "frames": track.get("frames", []),
            "weights": track.get("weights", session["dance_pack"].get("weights", [])),
            "events": track.get("events", (session["dance_pack"].get("events") or {}).get("events", [])),
            "coverage": float(track.get("coverage", 0.0)),
            "avg_x": float(track.get("avg_x", 0.5)),
        })

    session["coach_tracks"] = normalized
    session["coach_count"] = max(1, len(normalized))
    session["reference_loaded"] = True


def _register_player(session: Dict[str, Any], player_id: str, tracking_score: float = 0.0) -> Dict[str, Any]:
    player_id = str(player_id or "p0")
    player = session["players"].get(player_id)
    now_ms = int(time.time() * 1000)
    if player is None:
        slot = int(session["next_player_slot"])
        session["next_player_slot"] = slot + 1
        default_coach = slot % max(int(session.get("coach_count", 1)), 1)
        player = {
            "player_id": player_id,
            "slot": slot,
            "coach_index": default_coach,
            "ready": False,
            "tracking_score": float(tracking_score),
            "last_seen_ms": now_ms,
            "latest_pose": [],
        }
        session["players"][player_id] = player
        session["assignments"].setdefault(player_id, default_coach)
    else:
        player["tracking_score"] = float(tracking_score)
        player["last_seen_ms"] = now_ms
    return player


def _engine_for_player(session: Dict[str, Any], player_id: str) -> ScoringEngine:
    _ensure_reference_loaded(session)
    existing = session["player_engines"].get(player_id)
    coach_index = int(session["assignments"].get(player_id, 0))
    coach_index = max(0, min(coach_index, len(session["coach_tracks"]) - 1))

    if existing is not None and int(getattr(existing, "_danceflow_coach_index", coach_index)) == coach_index:
        return existing

    track = session["coach_tracks"][coach_index]
    pack = dict(session["dance_pack"])
    pack["reference_pose"] = {"frames": track.get("frames", [])}
    pack["weights"] = track.get("weights", [])
    pack["events"] = {"events": track.get("events", [])}
    engine = ScoringEngine(pack)
    engine.ref_frames = track.get("frames", [])
    setattr(engine, "_danceflow_coach_index", coach_index)
    if session.get("state") == "playing":
        engine.start()
    session["player_engines"][player_id] = engine
    return engine


def assign_coach(session: Dict[str, Any], player_id: str, coach_index: int) -> Dict[str, Any]:
    _ensure_reference_loaded(session)
    player = _register_player(session, player_id)
    coach_index = max(0, min(int(coach_index), max(len(session["coach_tracks"]) - 1, 0)))
    session["assignments"][player_id] = coach_index
    player["coach_index"] = coach_index

    old = session["player_engines"].pop(player_id, None)
    if old is not None:
        old.stop()
    if session.get("state") == "playing":
        _engine_for_player(session, player_id)
    return player


def public_players(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    now_ms = int(time.time() * 1000)
    players = []
    for player in sorted(session.get("players", {}).values(), key=lambda p: int(p.get("slot", 0))):
        players.append({
            "player_id": player.get("player_id", "p0"),
            "slot": int(player.get("slot", 0)),
            "coach_index": int(session.get("assignments", {}).get(player.get("player_id"), player.get("coach_index", 0))),
            "ready": bool(player.get("ready", False)),
            "tracking_score": round(float(player.get("tracking_score", 0.0)), 3),
            "active": now_ms - int(player.get("last_seen_ms", 0)) < 1500,
        })
    return players


async def _broadcast_players(session: Dict[str, Any]) -> None:
    desktop_ws = session.get("desktop_ws")
    if not desktop_ws:
        return
    try:
        await desktop_ws.send_json({
            "type": "players_changed",
            "players": public_players(session),
            "coach_count": int(session.get("coach_count", 1)),
        })
    except Exception:
        pass


async def _broadcast_score(session: Dict[str, Any], score_event) -> None:
    event_dict = score_event.model_dump() if hasattr(score_event, "model_dump") else score_event.dict()
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
                "player_id": score_event.player_id,
                "player_slot": score_event.player_slot,
                "coach_index": score_event.coach_index,
            })
        except Exception:
            pass


def _aggregate_results(session: Dict[str, Any]) -> Dict[str, Any]:
    player_results = []
    grade_counts = {"perfect": 0, "super": 0, "good": 0, "ok": 0, "x": 0}
    total_score = 0
    max_combo = 0
    arms, legs, torso = [], [], []

    for player_id, engine in session.get("player_engines", {}).items():
        result = engine.get_final_results()
        player = session["players"].get(player_id, {})
        entry = {
            "player_id": player_id,
            "player_slot": int(player.get("slot", 0)),
            "coach_index": int(session["assignments"].get(player_id, 0)),
            **result,
        }
        player_results.append(entry)
        total_score += int(result.get("total_score", 0))
        max_combo = max(max_combo, int(result.get("max_combo", 0)))
        for key in grade_counts:
            grade_counts[key] += int((result.get("grade_counts") or {}).get(key, 0))
        if result.get("accuracy_arms") is not None:
            arms.append(float(result["accuracy_arms"]))
            legs.append(float(result.get("accuracy_legs", 0.0)))
            torso.append(float(result.get("accuracy_torso", 0.0)))

    return {
        "total_score": total_score,
        "max_combo": max_combo,
        "grade_counts": grade_counts,
        "hold_results": [],
        "accuracy_arms": round(sum(arms) / max(len(arms), 1), 3) if arms else 0.0,
        "accuracy_legs": round(sum(legs) / max(len(legs), 1), 3) if legs else 0.0,
        "accuracy_torso": round(sum(torso) / max(len(torso), 1), 3) if torso else 0.0,
        "timeline_scores": [],
        "players": sorted(player_results, key=lambda p: p["player_slot"]),
    }


async def _finish_session(session: Dict[str, Any], reason: str = "completed"):
    if session.get("result_saved"):
        return _aggregate_results(session)

    for engine in session.get("player_engines", {}).values():
        engine.stop()
    session["state"] = "stopped"
    await db.update_session(session["session_id"], status="stopped")

    results = _aggregate_results(session)
    await _save_results(session["session_id"], session["dance_id"], results)
    session["result_saved"] = True

    for ws in (session.get("desktop_ws"), session.get("phone_ws")):
        if ws:
            try:
                await ws.send_json({"type": "game_over", "reason": reason, "results": results})
            except Exception:
                pass
    return results


def _pose_payload(msg: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
    received_at_ms = int(time.time() * 1000)
    one_way_ms = max(0, min(220, int(msg.get("estimated_latency_ms", 0) or 0)))
    media_capture_ms = max(0, int(session.get("last_media_ms", 0) or 0) - one_way_ms)
    return {
        "timestamp_ms": msg.get("timestamp_ms", received_at_ms),
        "received_at_ms": received_at_ms,
        "media_capture_ms": media_capture_ms,
        "network_latency_ms": one_way_ms,
        "landmarks": msg.get("landmarks", []),
        "world_landmarks": msg.get("world_landmarks", []),
        "tracking_score": msg.get("tracking_score", 0.0),
    }


async def _handle_player_pose(session: Dict[str, Any], pose_msg: Dict[str, Any]) -> None:
    player_id = str(pose_msg.get("player_id", "p0"))
    tracking_score = float(pose_msg.get("tracking_score", 0.0) or 0.0)
    is_new = player_id not in session["players"]
    player = _register_player(session, player_id, tracking_score)
    player["latest_pose"] = pose_msg.get("landmarks", [])

    if session.get("state") == "playing":
        engine = _engine_for_player(session, player_id)
        engine.add_pose_frame(_pose_payload(pose_msg, session))
    else:
        if is_new:
            session["calibrated"] = False
            session["state"] = "calibrating"
        buffers = session["calibration_frames"].setdefault(player_id, [])
        landmarks = pose_msg.get("landmarks", [])
        if landmarks:
            buffers.append(landmarks)
            if len(buffers) > 18:
                del buffers[:-18]
        if len(buffers) >= 10:
            try:
                result = run_calibration(buffers[-10:])
                result_dict = result.model_dump() if hasattr(result, "model_dump") else result.dict()
                player["ready"] = bool(result_dict.get("ready", False))
            except Exception:
                player["ready"] = tracking_score >= 0.55

        active = [p for p in session["players"].values() if int(time.time() * 1000) - int(p.get("last_seen_ms", 0)) < 1200]
        if active and all(bool(p.get("ready", False)) for p in active):
            session["calibrated"] = True
            session["state"] = "calibrated"
            await db.update_session(session["session_id"], status="calibrated")

    if is_new:
        await _broadcast_players(session)


@router.websocket("/ws/game/{session_id}")
async def game_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = active_sessions.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    session["desktop_ws"] = websocket
    await _broadcast_players(session)

    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            action = msg.get("action")

            if action == "start":
                await asyncio.to_thread(_ensure_reference_loaded, session)
                if not session["players"]:
                    _register_player(session, "p0", 1.0)
                for player_id in list(session["players"].keys()):
                    engine = _engine_for_player(session, player_id)
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
                await websocket.send_json({
                    "type": "started",
                    "server_timestamp_ms": int(time.time() * 1000),
                    "players": public_players(session),
                })

            elif action == "media_clock":
                media_ms = max(0, int(msg.get("media_time_ms", 0) or 0))
                session["last_media_ms"] = media_ms

                duration_ms = 0
                if session.get("player_engines"):
                    duration_ms = max((e.duration_ms for e in session["player_engines"].values()), default=0)
                if duration_ms and media_ms >= max(0, duration_ms - 20):
                    await _finish_session(session, reason="media_ended")
                    continue

                now_ms = int(time.time() * 1000)
                for player_id, engine in list(session.get("player_engines", {}).items()):
                    if not engine.is_running or engine.is_paused:
                        continue
                    score_event = engine.score_tick(media_ms, now_server_ms=now_ms)
                    if not score_event:
                        continue
                    player = session["players"].get(player_id, {})
                    score_event.player_id = player_id
                    score_event.player_slot = int(player.get("slot", 0))
                    score_event.coach_index = int(session["assignments"].get(player_id, 0))
                    score_event.player_pose = list(player.get("latest_pose", []))
                    await _broadcast_score(session, score_event)

            elif action == "pause":
                for engine in session.get("player_engines", {}).values():
                    engine.pause()
                session["state"] = "paused"
                await db.update_session(session_id, status="paused")
                if session.get("phone_ws"):
                    try:
                        await session["phone_ws"].send_json({"type": "pause"})
                    except Exception:
                        pass

            elif action == "resume":
                for engine in session.get("player_engines", {}).values():
                    engine.resume()
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
                session["calibration_frames"].clear()
                for p in session["players"].values():
                    p["ready"] = False
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

    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            msg_type = msg.get("type")

            if msg_type == "multi_pose_frame":
                poses = msg.get("poses") or []
                for pose in poses[:6]:
                    pose["timestamp_ms"] = msg.get("timestamp_ms", pose.get("timestamp_ms"))
                    pose["estimated_latency_ms"] = msg.get("estimated_latency_ms", pose.get("estimated_latency_ms", 0))
                    await _handle_player_pose(session, pose)

            elif msg_type in {"pose_frame", "calibration_frame"}:
                legacy = dict(msg)
                legacy["player_id"] = str(msg.get("player_id", "p0"))
                await _handle_player_pose(session, legacy)

            elif msg_type == "control" and msg.get("action") == "ready":
                session["calibrated"] = True
                for p in session["players"].values():
                    p["ready"] = True
                if session.get("state") != "playing":
                    session["state"] = "calibrated"
                    await db.update_session(session_id, status="calibrated")
                await _broadcast_players(session)

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
