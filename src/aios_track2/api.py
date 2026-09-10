from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .strategies import TOP5_STRATEGIES

REPO_ROOT = Path(__file__).resolve().parents[2]


def create_app(submission_dir: Path | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, JSONResponse
    except ImportError as exc:
        raise RuntimeError("install project with [api] extras") from exc

    from .control_room import default_submission_dir, load_control_room

    resolved = (submission_dir or default_submission_dir()).resolve()
    app = FastAPI(title="AIOS Track 2 Control Room", version="0.2.0")

    def room_payload():
        if not (resolved / "winner.json").exists() or not (resolved / "wells_schedule.inc").exists():
            raise HTTPException(status_code=404, detail=f"submission artifacts missing in {resolved}")
        return load_control_room(resolved)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "submission_dir": str(resolved)}

    @app.get("/api/strategies")
    def strategies():
        return [asdict(item) for item in TOP5_STRATEGIES]

    @app.get("/api/room")
    def room():
        return room_payload()

    @app.get("/api/recommendation")
    def recommendation():
        return room_payload()["recommendation"]

    @app.get("/download/wells_schedule.inc")
    def download_schedule():
        path = resolved / "wells_schedule.inc"
        if not path.exists():
            raise HTTPException(status_code=404, detail="wells_schedule.inc not packaged")
        return FileResponse(path, filename="wells_schedule.inc", media_type="text/plain")

    @app.get("/")
    def root():
        index = REPO_ROOT / "ui" / "index.html"
        if not index.exists():
            return JSONResponse({"error": "ui/index.html missing"}, status_code=500)
        return FileResponse(index)

    return app


app = None
try:
    app = create_app()
except RuntimeError:
    pass
