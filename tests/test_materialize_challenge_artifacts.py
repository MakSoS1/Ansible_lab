from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from materialize_challenge_artifacts import materialize_artifacts


def _write_artifact(root: Path, artifact_name: str) -> None:
    artifact = root / artifact_name
    artifact.mkdir(parents=True)
    (artifact / "scenario-manifest.json").write_text(artifact_name, encoding="utf-8")
    (artifact / "summary.npz").write_bytes(b"npz")


def test_scenario_artifacts_get_numeric_directories(tmp_path: Path) -> None:
    source = tmp_path / "download"
    destination = tmp_path / "runs"
    _write_artifact(source, "challenge-scenario-0")
    _write_artifact(source, "challenge-scenario-12")

    materialize_artifacts(source, destination, "scenario")

    assert (destination / "0" / "scenario-manifest.json").read_text(encoding="utf-8") == "challenge-scenario-0"
    assert (destination / "12" / "summary.npz").read_bytes() == b"npz"


def test_finalist_and_robustness_names_match_downstream_contract(tmp_path: Path) -> None:
    finalist_source = tmp_path / "finalist-download"
    finalists = tmp_path / "finalists"
    _write_artifact(finalist_source, "challenge-finalist-cem")
    materialize_artifacts(finalist_source, finalists, "finalist")
    assert (finalists / "cem" / "scenario-manifest.json").exists()

    robustness_source = tmp_path / "robustness-download"
    robustness = tmp_path / "robustness"
    _write_artifact(robustness_source, "challenge-robustness-mappo-2")
    materialize_artifacts(robustness_source, robustness, "robustness")
    assert (robustness / "mappo__perturb_2" / "summary.npz").exists()


def test_duplicate_target_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "download"
    destination = tmp_path / "runs"
    _write_artifact(source, "challenge-scenario-1")
    (destination / "1").mkdir(parents=True)

    with pytest.raises(FileExistsError):
        materialize_artifacts(source, destination, "scenario")
