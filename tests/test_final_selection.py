import pytest

from aios_track2.final_selection import VerifiedCandidate, choose_verified_winner, verify_clean_rerun


def candidate(
    name: str,
    npv: float,
    *,
    status: str = "success",
    max_wlpr: float = 100.0,
    schedule_sha: str = "abc",
    robustness: tuple[float, ...] = (),
) -> VerifiedCandidate:
    return VerifiedCandidate(
        name=name,
        vector=(1.0,) * 18,
        opm_npv_mrub=npv,
        max_wlpr=max_wlpr,
        status=status,
        schedule_sha256=schedule_sha,
        robustness_npvs_mrub=robustness,
    )


def test_winner_is_selected_by_real_opm_npv_after_hard_gates() -> None:
    winner = choose_verified_winner(
        [
            candidate("surrogate_favorite", 12_000.0, status="failed"),
            candidate("valid_lower", 11_900.0, robustness=(11_850.0, 11_860.0)),
            candidate("valid_higher", 11_950.0, robustness=(11_800.0, 11_810.0)),
            candidate("invalid_wlpr", 12_100.0, max_wlpr=501.0),
        ]
    )
    assert winner.name == "valid_higher"


def test_no_real_opm_candidate_fails_closed() -> None:
    with pytest.raises(ValueError, match="no real-OPM candidate"):
        choose_verified_winner([candidate("bad", 12_000.0, status="timeout")])


def test_clean_rerun_requires_same_schedule_npv_and_wlpr_gate() -> None:
    winner = candidate("winner", 11_950.25, schedule_sha="deadbeef")
    ok = verify_clean_rerun(
        winner,
        clean_status="success",
        clean_schedule_sha256="deadbeef",
        clean_npv_mrub=11_950.25,
        clean_max_wlpr=120.0,
    )
    assert ok["passed"] is True

    bad = verify_clean_rerun(
        winner,
        clean_status="success",
        clean_schedule_sha256="cafebabe",
        clean_npv_mrub=11_950.30,
        clean_max_wlpr=501.0,
        npv_abs_tolerance_mrub=1e-3,
    )
    assert bad["passed"] is False
    assert set(bad["failures"]) == {"SCHEDULE_SHA_MISMATCH", "NPV_MISMATCH", "WLPR_GT_500"}
