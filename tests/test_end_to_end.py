import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_smoke(out: Path, seed: int = 42) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "aios_track2.cli", "smoke", "--seed", str(seed), "--out", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_smoke_pipeline_is_reproducible(tmp_path: Path) -> None:
    first = _run_smoke(tmp_path / "a", seed=42)
    second = _run_smoke(tmp_path / "b", seed=42)
    assert first["sha"] == second["sha"]
    assert first["npv_mrub"] == second["npv_mrub"]
