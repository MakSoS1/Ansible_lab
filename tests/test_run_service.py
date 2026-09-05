from __future__ import annotations

import pytest

from aios_track2 import run_service

pytestmark = pytest.mark.skipif(
    not run_service.verified_run().exists(),
    reason="submission/ artifacts are not packaged",
)


def test_case_summary_describes_the_real_model_z_contract() -> None:
    case = run_service.case_summary()
    assert case["well_count"] == run_service.EXPECTED_WELLS
    assert tuple(case["dimensions"]) == run_service.EXPECTED_DIMENSIONS
    assert case["contract_start"].startswith("2007-01-01")
    assert case["wlpr_limit_m3_d"] == 500.0
    assert case["months"] == 225


def test_every_well_carries_a_role_and_a_control_group() -> None:
    wells = run_service.field_layout()
    assert len(wells) == run_service.EXPECTED_WELLS
    assert {well.role for well in wells} <= {"producer", "injector", "dual", "idle"}
    for well in wells:
        if well.role in {"producer", "dual"}:
            assert 0 <= well.producer_group < 4
        if well.role in {"injector", "dual"}:
            assert 0 <= well.injector_group < 2


@pytest.mark.skipif(not run_service.verified_run().has_baseline(), reason="baseline economics is not packaged")
def test_headline_metrics_beat_the_baseline_and_respect_the_liquid_limit() -> None:
    metrics = run_service.headline_metrics()
    assert metrics["npv_mrub"] > metrics["baseline_npv_mrub"]
    assert metrics["delta_mrub"] == pytest.approx(metrics["npv_mrub"] - metrics["baseline_npv_mrub"], abs=1e-6)
    assert metrics["max_wlpr_m3_d"] <= run_service.WLPR_LIMIT_M3_D
    assert metrics["opm_calls"]["total"] == metrics["opm_calls"]["training"] + 18 + 1


def test_constraint_guard_accepts_the_submitted_policy() -> None:
    assert run_service.policy_explanation()["bounds"]["passed"] is True


def test_constraint_guard_rejects_a_policy_outside_the_box() -> None:
    report = run_service.check_policy_bounds([1.5] * (run_service.CHALLENGE_GROUPS * 3))
    assert report["passed"] is False
    assert report["out_of_bounds"]


def test_constraint_guard_rejects_a_step_larger_than_the_node_limit() -> None:
    vector = [1.0, 1.0, 1.0] * run_service.CHALLENGE_GROUPS
    vector[1] = 1.2
    report = run_service.check_policy_bounds(vector)
    assert report["passed"] is False
    assert report["delta_violations"]


def test_surrogate_quality_shows_the_failed_top3_gate_instead_of_hiding_it() -> None:
    quality = run_service.surrogate_quality()
    failed = [gate["label"] for gate in quality["gates"] if not gate["passed"]]
    assert failed == ["Полнота top-3"]
    assert quality["holdout_passed"] is False
    assert quality["tournament_authorized"] is True
    assert quality["channels"][0]["r2"] >= quality["channels"][-1]["r2"]


@pytest.mark.skipif(not run_service.verified_run().has_baseline(), reason="baseline economics is not packaged")
def test_production_series_aligns_baseline_and_winner() -> None:
    series = run_service.production_series()
    assert len(series["winner"]["oil_t"]) == len(series["baseline"]["oil_t"]) == 225
    assert sum(series["winner"]["oil_t"]) > sum(series["baseline"]["oil_t"])


def test_verification_run_passes_every_gate_and_unlocks_the_download() -> None:
    state = run_service.REGISTRY.create("verify")
    run_service.REGISTRY.execute(state)
    payload = state.as_dict()
    assert payload["state"] == "VERIFIED", payload.get("error")
    assert payload["downloadable"] is True
    assert [event["step"] for event in payload["events"]] == list(range(1, 10))
    assert not [event for event in payload["events"] if event["status"] == "fail"]
    assert payload["result"]["schedule_sha256"] == run_service.verified_run().manifest["schedule_sha256"]


def test_agent_messages_use_russian_decimal_separators() -> None:
    state = run_service.REGISTRY.create("verify")
    run_service.REGISTRY.execute(state)
    economics = next(event for event in state.events if event.agent == "Economics")
    assert "12 475,95" in economics.message.replace(" ", " ")


def test_unknown_run_mode_is_refused() -> None:
    with pytest.raises(ValueError):
        run_service.REGISTRY.create("teleport")


@pytest.mark.skipif(not run_service.model_z_archive().exists(), reason="Model Z archive is not available")
def test_schedule_rebuilt_from_the_untouched_deck_reproduces_the_submitted_hash() -> None:
    manifest = run_service.verified_run().manifest
    text, facts = run_service.regenerate_schedule([float(value) for value in manifest["winner"]["vector"]])
    assert facts["sha256"] == manifest["schedule_sha256"]
    assert facts["well_count"] == run_service.EXPECTED_WELLS
    assert tuple(facts["dimensions"]) == run_service.EXPECTED_DIMENSIONS
    assert facts["history_prefix_identical"] is True
    assert text.startswith("RPTSCHED")
