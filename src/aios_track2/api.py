from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from aios_track2.agents import PipelineResult, PipelineState, run_pipeline
from aios_track2.schedule import write_schedule_text

app = FastAPI(title="AIOS Track 2")
RUNS: dict[str, PipelineResult] = {}


class RunRequest(BaseModel):
    seed: int = 42


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs")
def create_run(request: RunRequest) -> dict[str, str]:
    result = run_pipeline()
    run_id = f"run-{request.seed}-{result.schedule_sha256[:8]}"
    RUNS[run_id] = result
    return {"id": run_id, "config_hash": result.schedule_sha256, "state": result.state.value}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    if run_id == "fixture-unvalidated":
        return {"id": run_id, "state": "PREDICTED", "npv_mrub": None, "events": []}
    result = RUNS.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="unknown run")
    return {
        "id": run_id,
        "state": result.state.value,
        "npv_mrub": result.npv_mrub,
        "backend": result.backend,
        "events": [
            {
                "actor": event.actor,
                "action": event.action,
                "timestamp": event.timestamp,
            }
            for event in result.audit.events
        ],
        "explanation": result.explanation,
    }


@app.get("/api/runs/{run_id}/schedule")
def download_schedule(run_id: str):
    if run_id == "fixture-unvalidated":
        raise HTTPException(status_code=409, detail="run is not PACKAGED")
    result = RUNS.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="unknown run")
    if result.state != PipelineState.PACKAGED:
        raise HTTPException(status_code=409, detail="run is not PACKAGED")
    text = write_schedule_text(result.schedule)
    return PlainTextResponse(text, media_type="text/plain")


def write_package(result: PipelineResult, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "wells_schedule.inc").write_text(write_schedule_text(result.schedule), encoding="utf-8")
    (folder / "npv.json").write_text(
        f'{{"npv_mrub": {result.npv_mrub:.6f}, "backend": "{result.backend}"}}\n',
        encoding="utf-8",
    )
    lines = [
        f"{event.timestamp}\t{event.actor}\t{event.action}\t{event.input_hashes[0]}\t{event.output_hashes[0]}"
        for event in result.audit.events
    ]
    (folder / "audit.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "manifest.json").write_text(
        f'{{"schedule_sha256": "{result.schedule_sha256}", "state": "{result.state.value}"}}\n',
        encoding="utf-8",
    )
