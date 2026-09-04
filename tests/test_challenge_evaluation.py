from aios_track2.challenge_evaluation import physical_inventory_gate


def manifest(*, baseline=False, changed=None, wlpr=100.0, status="success"):
    return {
        "status": status,
        "compact_summary": {"max_wlpr": wlpr},
        "summary_install_changed_files": changed or ["Model_Z/Model_Z_summary.inc"],
        "baseline_schedule_byte_identical": baseline,
    }


def test_physical_gate_accepts_nested_summary_path_only() -> None:
    report = physical_inventory_gate(
        {0: manifest(baseline=True), 1: manifest()},
        expected_ids={0, 1},
    )
    assert report["passed"] is True


def test_physical_gate_rejects_missing_extra_physical_change_and_wlpr() -> None:
    report = physical_inventory_gate(
        {
            0: manifest(baseline=True),
            1: manifest(changed=["Model_Z/Model_Z_summary.inc", "Model_Z/Model_Z_grid.inc"], wlpr=501.0),
            3: manifest(),
        },
        expected_ids={0, 1, 2},
    )
    assert report["passed"] is False
    assert report["missing_ids"] == [2]
    assert report["extra_ids"] == [3]
    assert report["wlpr_violations"] == [1]
    assert report["unexpected_telemetry_mutations"] == [1]
