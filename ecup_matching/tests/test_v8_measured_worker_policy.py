from pathlib import Path


def test_predict_v6_uses_measured_eight_worker_policy_for_full_runtime():
    source = Path("ecup_matching/submission/predict_v6.py").read_text(encoding="utf-8")
    assert "structured_workers = resolve_worker_count(8)" in source
    assert "workers=structured_workers" in source


def test_measured_worker_policy_is_shared_by_structured_and_dual_text_cache():
    source = Path("ecup_matching/submission/predict_v6.py").read_text(encoding="utf-8")
    # One resolved worker count is deliberately reused for both prediction-preserving CPU phases.
    assert source.count("workers=structured_workers") >= 2
