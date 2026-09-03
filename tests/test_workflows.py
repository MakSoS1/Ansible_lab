from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow_yaml(relative: str) -> dict:
    payload = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    if True in payload:
        payload["on"] = payload.pop(True)
    return payload


def workflow_text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_training_is_never_triggered_by_pull_request() -> None:
    training = workflow_yaml(".github/workflows/aios-train-surrogate.yml")
    assert set(training["on"]) == {"workflow_dispatch"} or list(training["on"].keys()) == ["workflow_dispatch"]


def test_hf_token_is_referenced_as_secret() -> None:
    import re

    text = workflow_text(".github/workflows/aios-publish-results.yml")
    assert "secrets.HF_TOKEN" in text
    assert re.search(r"hf_[A-Za-z0-9]{8,}", text) is None


def test_training_uses_apple_silicon_runner() -> None:
    training = workflow_yaml(".github/workflows/aios-train-surrogate.yml")
    job = training["jobs"]["train"]
    assert job["runs-on"] == "macos-14"
