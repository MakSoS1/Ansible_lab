from __future__ import annotations

import sys
import types
from pathlib import Path

from aios_track2.hfstore import RunManifest, upload_run_directory


class _FakeHfApi:
    calls: list[tuple[str, dict[str, object]]] = []

    def __init__(self, *, token: str) -> None:
        self.calls.append(("init", {"token": token}))

    def create_repo(self, **kwargs: object) -> object:
        self.calls.append(("create_repo", dict(kwargs)))
        return types.SimpleNamespace(repo_id=kwargs["repo_id"])

    def upload_folder(self, **kwargs: object) -> object:
        self.calls.append(("upload_folder", dict(kwargs)))
        return types.SimpleNamespace(oid="fake")


def test_upload_creates_private_dataset_before_upload(monkeypatch, tmp_path: Path) -> None:
    _FakeHfApi.calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=_FakeHfApi))

    manifest = RunManifest(
        run_id="final-mappo",
        git_sha="9ad40738134d1fc0beb1df62214d09bc9ea8d114",
        github_run_id="33925326455",
        dataset_id="MakSoS1/aios-track2-runs",
        seed=9200,
        simulator_version="OPM Flow 2026.04",
        deck_sha256="a" * 64,
        schedule_sha256="b" * 64,
        status="verified",
        npv_rub=12_475_954_558.553085,
    )

    upload_run_directory(tmp_path, manifest, token="secret-token")

    assert [name for name, _ in _FakeHfApi.calls] == ["init", "create_repo", "upload_folder"]
    assert _FakeHfApi.calls[1][1] == {
        "repo_id": "MakSoS1/aios-track2-runs",
        "repo_type": "dataset",
        "private": True,
        "exist_ok": True,
    }
    assert _FakeHfApi.calls[2][1] == {
        "repo_id": "MakSoS1/aios-track2-runs",
        "repo_type": "dataset",
        "folder_path": str(tmp_path),
        "path_in_repo": "runs/9ad40738134d-33925326455/final-mappo",
        "commit_message": "track2 run final-mappo",
    }
