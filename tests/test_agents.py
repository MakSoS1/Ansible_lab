from aios_track2.agents import Orchestrator


def test_explanation_agent_cannot_modify_schedule() -> None:
    result = Orchestrator().run_fixture()
    assert result.schedule_sha256 == result.audit.schedule_before_explanation_sha256


def test_economics_is_only_npv_writer() -> None:
    result = Orchestrator().run_fixture()
    assert result.audit.writers_for("npv_mrub") == {"EconomicsAgent"}
