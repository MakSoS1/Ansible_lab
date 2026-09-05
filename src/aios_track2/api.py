"""HTTP surface of the Track 2 control room.

Read-only apart from starting a demonstration run: control decisions belong to
the pipeline, never to the browser. Every payload is derived from the packaged
submission in ``submission/`` or recomputed from the untouched Model Z deck.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import run_service
from .strategies import TOP5_STRATEGIES

REPO_ROOT = Path(__file__).resolve().parents[2]


def ui_directory() -> Path:
    return REPO_ROOT / "ui"


def create_app(submission_dir: Path | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - only without the [api] extra
        raise RuntimeError("install project with [api] extras: pip install -e '.[api]'") from exc

    from .control_room import default_submission_dir, load_control_room

    resolved = (submission_dir or default_submission_dir()).resolve()
    run_service.use_submission_dir(resolved)
    app = FastAPI(title="AIOS Track 2 Control Room", version="0.3.0")

    def require_artifacts() -> None:
        if not run_service.verified_run().exists():
            raise HTTPException(status_code=404, detail=f"submission artifacts missing in {resolved}")

    def room_payload() -> dict[str, Any]:
        require_artifacts()
        return load_control_room(resolved)

    # ---------------------------------------------------------------- status
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        run = run_service.verified_run()
        return {
            "status": "ok",
            "submission_dir": str(resolved),
            "artifacts": run.exists(),
            "baseline": run.has_baseline(),
            "model_z_archive": run_service.model_z_archive().exists(),
        }

    @app.get("/api/strategies")
    def strategies() -> list[dict[str, Any]]:
        return [asdict(item) for item in TOP5_STRATEGIES]

    # ------------------------------------------------------------- read model
    @app.get("/api/case")
    def case() -> dict[str, Any]:
        require_artifacts()
        return run_service.case_summary()

    @app.get("/api/metrics")
    def metrics() -> dict[str, Any]:
        require_artifacts()
        return run_service.headline_metrics()

    @app.get("/api/field")
    def field() -> dict[str, Any]:
        require_artifacts()
        return {"wells": [well.as_dict() for well in run_service.field_layout()]}

    @app.get("/api/production")
    def production() -> dict[str, Any]:
        require_artifacts()
        return run_service.production_series()

    @app.get("/api/annual")
    def annual() -> dict[str, Any]:
        require_artifacts()
        return {"rows": run_service.annual_economics()}

    @app.get("/api/operations")
    def operations() -> dict[str, Any]:
        require_artifacts()
        return run_service.well_operations()

    @app.get("/api/quality")
    def quality() -> dict[str, Any]:
        require_artifacts()
        return run_service.surrogate_quality()

    @app.get("/api/explanation")
    def explanation() -> dict[str, Any]:
        require_artifacts()
        return run_service.policy_explanation()

    @app.get("/api/room")
    def room() -> dict[str, Any]:
        return room_payload()

    @app.get("/api/recommendation")
    def recommendation() -> dict[str, Any]:
        return room_payload()["recommendation"]

    # -------------------------------------------------------------- run cycle
    @app.post("/api/runs", status_code=202)
    def start_run(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        require_artifacts()
        mode = str((payload or {}).get("mode", "verify"))
        if mode == "rebuild" and not run_service.model_z_archive().exists():
            raise HTTPException(status_code=409, detail="Model Z archive is not shipped, rebuild mode is unavailable")
        try:
            return run_service.REGISTRY.start(mode).as_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _run_or_404(run_id: str) -> run_service.RunState:
        state = run_service.REGISTRY.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="unknown run")
        return state

    @app.get("/api/runs/{run_id}")
    def read_run(run_id: str) -> dict[str, Any]:
        return _run_or_404(run_id).as_dict()

    @app.get("/api/runs/{run_id}/report.json")
    def run_report(run_id: str) -> JSONResponse:
        state = _run_or_404(run_id)
        return JSONResponse(
            state.as_dict(),
            headers={"Content-Disposition": f'attachment; filename="aios-track2-run-{run_id}.json"'},
        )

    @app.get("/api/runs/{run_id}/schedule")
    def run_schedule(run_id: str) -> FileResponse:
        state = _run_or_404(run_id)
        if state.state != "VERIFIED":
            raise HTTPException(status_code=409, detail=f"schedule stays locked while the run is {state.state}")
        return FileResponse(
            run_service.verified_run().schedule_path,
            media_type="text/plain; charset=utf-8",
            filename="wells_schedule.inc",
        )

    # ------------------------------------------------------------- deliverable
    @app.get("/download/wells_schedule.inc")
    def download_schedule() -> FileResponse:
        path = resolved / "wells_schedule.inc"
        if not path.exists():
            raise HTTPException(status_code=404, detail="wells_schedule.inc not packaged")
        return FileResponse(path, filename="wells_schedule.inc", media_type="text/plain; charset=utf-8")

    # -------------------------------------------------------------------- ui
    ui = ui_directory()
    if ui.exists():
        app.mount("/static", StaticFiles(directory=ui), name="static")

    @app.get("/", include_in_schema=False)
    def root():
        index = ui / "index.html"
        if not index.exists():
            return JSONResponse({"error": "ui/index.html missing"}, status_code=500)
        return FileResponse(index)

    return app


app = None
try:
    app = create_app()
except RuntimeError:  # pragma: no cover - only without the [api] extra
    pass
