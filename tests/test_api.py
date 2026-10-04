from fastapi.testclient import TestClient

from aios_track2.api import app

client = TestClient(app)


def test_schedule_endpoint_refuses_unvalidated_run() -> None:
    response = client.get("/api/runs/fixture-unvalidated/schedule")
    assert response.status_code == 409


def test_validated_schedule_is_downloadable() -> None:
    created = client.post("/api/runs", json={"seed": 42})
    assert created.status_code == 200
    run_id = created.json()["id"]
    response = client.get(f"/api/runs/{run_id}/schedule")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}
