from aios_track2.challenge_evaluation import physical_inventory_gate, real_opm_tournament_gate


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


def _evaluation(*, top_k=2 / 3, simple_regret=0.0, physical=True) -> dict:
    return {
        "passed": False,
        "failures": ["NPV_TOP3_RECALL_LT_090"],
        "physical_gate": {"passed": physical},
        "reference_parity": {"passed": True},
        "dynamic_selection": {
            "holdout": {
                "min_aggregate_channel_r2": 0.9512247034582642,
                "max_aggregate_channel_nrmse": 0.027523284453274358,
            }
        },
        "npv_selection": {
            "holdout": {
                "spearman": 0.9911764705882352,
                "pairwise_accuracy": 0.975,
                "top_k_recall": top_k,
                "simple_regret": simple_regret,
            }
        },
    }


def test_top3_miss_stays_audited_but_does_not_block_real_opm_tournament() -> None:
    report = real_opm_tournament_gate(_evaluation())
    assert report["passed"] is True
    assert report["surrogate_holdout_passed"] is False
    assert report["surrogate_holdout_failures"] == ["NPV_TOP3_RECALL_LT_090"]
    assert report["audited_top_k_recall"] == 2 / 3


def test_real_opm_tournament_rejects_wrong_selected_best() -> None:
    report = real_opm_tournament_gate(_evaluation(simple_regret=1.0))
    assert report["passed"] is False
    assert "NPV_SIMPLE_REGRET_GT_0" in report["failures"]


def test_real_opm_tournament_rejects_failed_physics() -> None:
    report = real_opm_tournament_gate(_evaluation(physical=False))
    assert report["passed"] is False
    assert "PHYSICAL_GATE_FAILED" in report["failures"]
