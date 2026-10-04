from pathlib import Path

from aios_track2.opm import FlowRequest, run_flow


def test_flow_result_is_hashed(tmp_path: Path) -> None:
    result = run_flow(FlowRequest(deck=Path("tests/fixtures/minimal.DATA"), output_dir=tmp_path, timeout_seconds=10))
    assert result.status == "success"
    assert len(result.stdout_sha256) == 64
