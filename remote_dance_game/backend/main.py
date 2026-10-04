import os
import socket
import asyncio
import copy
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import qrcode
import io
import base64

from config import (
    HOST, PORT, DANCES_DIR, DATA_DIR, BASE_DIR, PUBLIC_BASE_URL,
    SCORING_THRESHOLDS, SCORING_WEIGHTS, MATCHING_WINDOW_MS,
)
import config
import database as db
from api.dances import router as dances_router
from api.ws import router as ws_router, create_game_session, active_sessions, assign_coach, public_players
from models.schemas import GameSessionCreate


dance_cache: dict = {}

DEFAULT_APP_SETTINGS = {
    "scoring_thresholds_base": copy.deepcopy(SCORING_THRESHOLDS),
    "scoring_weights": copy.deepcopy(SCORING_WEIGHTS),
    "matching_window_ms": int(MATCHING_WINDOW_MS),
    "sensitivity": 1.0,
}


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _sanitize_thresholds(raw: Dict[str, Any], fallback: Dict[str, float]) -> Dict[str, float]:
    out = copy.deepcopy(fallback)
    for key in out.keys():
        if key in raw:
            try:
                out[key] = float(raw[key])
            except Exception:
                pass
    return out


def _sanitize_weights(raw: Dict[str, Any], fallback: Dict[str, float]) -> Dict[str, float]:
    out = copy.deepcopy(fallback)
    for key in out.keys():
        if key in raw:
            try:
                out[key] = float(raw[key])
            except Exception:
                pass
    return out


def _effective_thresholds(base_thresholds: Dict[str, float], sensitivity: float) -> Dict[str, float]:
    effective = {}
    for key, value in base_thresholds.items():
        effective[key] = round(_clamp(float(value) * sensitivity, 0.3, 0.99), 4)
    return effective


async def _load_runtime_settings() -> Dict[str, Any]:
    stored = await db.get_all_settings()
    base_thresholds = _sanitize_thresholds(
        stored.get("scoring_thresholds_base", {}),
        DEFAULT_APP_SETTINGS["scoring_thresholds_base"],
    )
    weights = _sanitize_weights(
        stored.get("scoring_weights", {}),
        DEFAULT_APP_SETTINGS["scoring_weights"],
    )
    sensitivity = _clamp(float(stored.get("sensitivity", DEFAULT_APP_SETTINGS["sensitivity"])), 0.5, 1.5)
    matching_window_ms = int(stored.get("matching_window_ms", DEFAULT_APP_SETTINGS["matching_window_ms"]))
    matching_window_ms = int(_clamp(matching_window_ms, 60, 250))

    effective_thresholds = _effective_thresholds(base_thresholds, sensitivity)
    config.SCORING_THRESHOLDS.update(effective_thresholds)
    config.SCORING_WEIGHTS.update(weights)
    config.MATCHING_WINDOW_MS = matching_window_ms

    return {
        "scoring_thresholds_base": base_thresholds,
        "scoring_thresholds": effective_thresholds,
        "scoring_weights": weights,
        "matching_window_ms": matching_window_ms,
        "sensitivity": sensitivity,
    }


async def _persist_runtime_settings(settings: Dict[str, Any]):
    await db.set_setting("scoring_thresholds_base", settings["scoring_thresholds_base"])
    await db.set_setting("scoring_weights", settings["scoring_weights"])
    await db.set_setting("matching_window_ms", settings["matching_window_ms"])
    await db.set_setting("sensitivity", settings["sensitivity"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DANCES_DIR, exist_ok=True)
    await db.init_db()
    app.state.runtime_settings = await _load_runtime_settings()
    await _persist_runtime_settings(app.state.runtime_settings)
    yield


app = FastAPI(title="Dance Coach Game", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dances_router)
app.include_router(ws_router)

mobile_dir = os.path.join(BASE_DIR, "mobile")
if os.path.isdir(mobile_dir):
    app.mount("/mobile", StaticFiles(directory=mobile_dir, html=True), name="mobile")

desktop_dist_dir = os.path.join(BASE_DIR, "desktop", "dist")
if os.path.isdir(desktop_dist_dir):
    # Optional browser mode: after `cd desktop && npm run build`, FastAPI can
    # serve the desktop UI too. This is the cleanest mode for HTTPS tunnels:
    # open https://<tunnel>/app/, scan QR, phone opens the same HTTPS origin.
    app.mount("/app", StaticFiles(directory=desktop_dist_dir, html=True), name="desktop_app")


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _external_base_url(request: Request) -> str:
    """Return the base URL that should be opened from the phone.

    Phone browser camera access needs a secure context.
    For phone-browser mode set PUBLIC_BASE_URL to an HTTPS tunnel/domain
    pointing to this backend (for example Cloudflare Tunnel).
    """
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL

    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host = request.headers.get("x-forwarded-host")
    if forwarded_proto and forwarded_host:
        return f"{forwarded_proto}://{forwarded_host}".rstrip("/")

    host_header = request.headers.get("host", f"127.0.0.1:{PORT}")
    host_only = host_header.split(":", 1)[0].lower()

    # If desktop talks to backend through localhost, phone cannot use that URL.
    # Fallback to LAN IP for direct-LAN tests.
    if host_only in {"localhost", "127.0.0.1", "::1"}:
        return f"http://{_get_local_ip()}:{PORT}"

    return f"http://{host_header}".rstrip("/")


@app.get("/")
async def root():
    return {"name": "Dance Coach Game", "version": "1.0.0", "status": "running"}


@app.post("/api/session/create")
async def create_session(req: GameSessionCreate, request: Request):
    try:
        loop = asyncio.get_event_loop()
        session = await loop.run_in_executor(None, create_game_session, req.dance_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session error: {e}")

    base_url = _external_base_url(request)
    connect_url = f"{base_url}/mobile?session={session['session_id']}"

    qr_base64 = ""
    try:
        qr = qrcode.QRCode(box_size=6, border=1)
        qr.add_data(connect_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="white", back_color="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        qr_base64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    except Exception as e:
        print(f"QR generation failed: {e}")

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    await db.insert_session({
        "session_id": session["session_id"],
        "dance_id": req.dance_id,
        "status": "created",
        "created_at": now,
    })

    return {
        "session_id": session["session_id"],
        "dance_id": req.dance_id,
        "connect_url": connect_url,
        "qr_data": qr_base64,
    }


@app.get("/api/session/{session_id}/status")
async def session_status(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        return {"status": "not_found"}
    return {
        "status": session.get("state", "created"),
        "phone_connected": session.get("phone_connected", False),
        "calibrated": session.get("calibrated", False),
        "coach_count": int(session.get("coach_count", 1)),
        "players": public_players(session),
    }


@app.post("/api/session/{session_id}/assign")
async def session_assign(session_id: str, payload: dict):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    player_id = str(payload.get("player_id", "") or "")
    if not player_id:
        raise HTTPException(status_code=400, detail="player_id is required")
    try:
        coach_index = int(payload.get("coach_index", 0))
    except Exception:
        raise HTTPException(status_code=400, detail="coach_index must be an integer")
    player = assign_coach(session, player_id, coach_index)
    return {
        "status": "assigned",
        "player_id": player_id,
        "player_slot": int(player.get("slot", 0)),
        "coach_index": int(player.get("coach_index", 0)),
        "players": public_players(session),
    }


@app.get("/api/results/{session_id}")
async def get_results(session_id: str):
    result = await db.get_result(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")
    return result


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "dance-coach-game",
        "active_sessions": len(active_sessions),
    }


@app.get("/api/diagnostics/live")
async def diagnostics_live():
    sessions = []
    for sid, s in active_sessions.items():
        sessions.append({
            "session_id": sid,
            "dance_id": s.get("dance_id"),
            "state": s.get("state"),
            "phone_connected": s.get("phone_connected", False),
            "calibrated": s.get("calibrated", False),
            "pose_loaded": s.get("pose_loaded", False),
            "has_latest_score": bool(s.get("latest_score")),
            "coach_count": int(s.get("coach_count", 1)),
            "players": public_players(s),
        })
    return {
        "active_sessions": sessions,
        "settings": getattr(app.state, "runtime_settings", {}),
    }


@app.get("/video/{dance_id}")
async def serve_video(dance_id: str):
    d = await db.get_dance(dance_id)
    if not d:
        raise HTTPException(status_code=404, detail="Dance not found")
    video_path = d.get("video_path", "")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/audio/{dance_id}")
async def serve_audio(dance_id: str):
    d = await db.get_dance(dance_id)
    if not d:
        raise HTTPException(status_code=404, detail="Dance not found")
    audio_path = d.get("audio_path", "")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(audio_path, media_type="audio/mpeg")


@app.get("/api/settings")
async def get_settings():
    runtime_settings = await _load_runtime_settings()
    app.state.runtime_settings = runtime_settings
    return runtime_settings


@app.post("/api/settings")
async def update_settings(settings: dict):
    current = await _load_runtime_settings()

    base_thresholds = copy.deepcopy(current["scoring_thresholds_base"])
    weights = copy.deepcopy(current["scoring_weights"])
    sensitivity = float(current["sensitivity"])
    matching_window_ms = int(current["matching_window_ms"])

    if "scoring_thresholds" in settings:
        base_thresholds = _sanitize_thresholds(settings["scoring_thresholds"], base_thresholds)
    if "sensitivity" in settings:
        try:
            sensitivity = float(settings["sensitivity"])
        except Exception:
            pass
    sensitivity = _clamp(sensitivity, 0.5, 1.5)

    if "scoring_weights" in settings:
        weights = _sanitize_weights(settings["scoring_weights"], weights)
    if "matching_window_ms" in settings:
        try:
            matching_window_ms = int(settings["matching_window_ms"])
        except Exception:
            pass
    matching_window_ms = int(_clamp(matching_window_ms, 60, 250))

    updated = {
        "scoring_thresholds_base": base_thresholds,
        "scoring_thresholds": _effective_thresholds(base_thresholds, sensitivity),
        "scoring_weights": weights,
        "matching_window_ms": matching_window_ms,
        "sensitivity": sensitivity,
    }
    await _persist_runtime_settings(updated)
    app.state.runtime_settings = updated
    return {"status": "updated", **updated}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
