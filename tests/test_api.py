from __future__ import annotations

import time

import pytest

from aios_track2 import run_service
from aios_track2.api import create_app

fastapi_testclient = pytest.importorskip("fastapi.testclient")

pytestmark = pytest.mark.skipif(
    not run_service.verified_run().exists(),
    reason="submission/ artifacts are not packaged",
)


@pytest.fixture()
def client():
    with fastapi_testclient.TestClient(create_app()) as instance:
        yield instance


def _wait_for(client, run_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get(f"/api/runs/{run_id}").json()
        if payload["state"] not in {"QUEUED", "RUNNING"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("run did not finish in time")


def test_health_reports_whether_the_artifacts_are_installed(client) -> None:
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["artifacts"] is True


def test_case_endpoint_serves_the_deck_identity(client) -> None:
    payload = client.get("/api/case").json()
    assert payload["well_count"] == 103
    assert payload["deck_sha256"]


def test_field_endpoint_serves_every_well(client) -> None:
    assert len(client.get("/api/field").json()["wells"]) == 103


def test_legacy_control_room_routes_still_answer(client) -> None:
    assert client.get("/api/room").status_code == 200
    assert client.get("/api/recommendation").status_code == 200
    assert client.get("/download/wells_schedule.inc").status_code == 200


def test_run_completes_and_serves_the_schedule(client) -> None:
    created = client.post("/api/runs", json={"mode": "verify"})
    assert created.status_code == 202
    run_id = created.json()["run_id"]
    payload = _wait_for(client, run_id)
    assert payload["state"] == "VERIFIED", payload.get("error")
    download = client.get(f"/api/runs/{run_id}/schedule")
    assert download.status_code == 200
    assert download.text.startswith("RPTSCHED")


def test_schedule_is_locked_for_an_unfinished_run(client) -> None:
    state = run_service.REGISTRY.create("verify")
    assert client.get(f"/api/runs/{state.run_id}/schedule").status_code == 409


def test_unknown_run_is_a_404(client) -> None:
    assert client.get("/api/runs/does-not-exist").status_code == 404


def test_index_and_assets_are_served(client) -> None:
    index = client.get("/")
    assert index.status_code == 200
    assert "Model Z" in index.text
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
