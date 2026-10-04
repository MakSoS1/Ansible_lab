import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNITY = ROOT / "unity"

def test_unity_project_baseline():
    version = (UNITY / "ProjectSettings" / "ProjectVersion.txt").read_text(encoding="utf-8")
    assert "6000.3" in version
    deps = json.loads((UNITY / "Packages" / "manifest.json").read_text(encoding="utf-8"))["dependencies"]
    assert deps["com.unity.inputsystem"].startswith("1.20")
    assert "com.unity.render-pipelines.universal" in deps
    assert "com.unity.ugui" in deps

def test_unity_runtime_contracts_present():
    runtime = (UNITY / "Assets" / "Scripts" / "DanceFlowRuntime.cs").read_text(encoding="utf-8")
    for token in ["class DanceFlowApp", "class DanceApiClient", "class GameSocketClient", "class LibraryScreen", "class ConnectScreen", "class CoachSelectScreen", "class GameplayScreen", "class CoachStage3D", "class HumanoidPoseDriver", "class PosePreviewGraphic", "player_pose", "coach_cues", "CoachPreviewUrl", "3840", "pose-timeline"]:
        assert token in runtime

def test_editor_bootstrap_creates_main_scene():
    bootstrap = (UNITY / "Assets" / "Editor" / "ProjectBootstrap.cs").read_text(encoding="utf-8")
    assert "DanceFlowMain.unity" in bootstrap
    assert "AddComponent<DanceFlowApp>" in bootstrap
    assert "defaultScreenWidth = 3840" in bootstrap

def test_backend_exposes_pose_timeline():
    api = (ROOT / "backend" / "api" / "dances.py").read_text(encoding="utf-8")
    assert "pose-timeline" in api
    assert "world_landmarks" in api
    assert "build_pictogram_markers" in api
    assert "/coaches/{coach_index}/preview" in api
