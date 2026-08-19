from __future__ import annotations

import pandas as pd
import pytest

from ecup_matching.ml.v20_teacher_bakeoff import (
    score_pair,
    score_teacher,
    select_teacher_pair,
)


def _truth(n_each: int = 1200) -> pd.DataFrame:
    rows = []
    for i in range(n_each):
        rows.append({"id1": i, "id2": 100000 + i, "target": 1, "reason_code": "SAME_MODEL", "category": "phones"})
    for i in range(n_each):
        rows.append({"id1": 200000 + i, "id2": 300000 + i, "target": 0, "reason_code": "MODEL_CONFLICT", "category": "phones"})
    return pd.DataFrame(rows)


def _labels(truth: pd.DataFrame, *, mistakes: int = 0, invalid: int = 0) -> pd.DataFrame:
    out = truth[["id1", "id2", "target", "reason_code"]].rename(columns={"target": "pred"}).copy()
    out["valid"] = True
    out["uncertain"] = False
    if mistakes:
        idx = list(out.index[:mistakes])
        out.loc[idx, "pred"] = 1 - out.loc[idx, "pred"].astype(int)
    if invalid:
        out.loc[out.index[:invalid], "valid"] = False
    return out


def _runtime(model_id: str, family: str, *, vram: float = 5.0, rps: float = 5.0) -> dict:
    return {
        "model_id": model_id,
        "revision": "a" * 40,
        "family": family,
        "backend": "transformers",
        "quantization": "q4",
        "peak_vram_gib": vram,
        "rows_per_second": rps,
    }


def test_high_quality_teacher_is_eligible():
    truth = _truth()
    report = score_teacher(truth, _labels(truth), _runtime("Qwen/Qwen3.5-4B", "qwen35"))
    assert report["eligible"] is True
    assert report["json_valid_rate"] == pytest.approx(1.0)
    assert report["coverage"] == pytest.approx(1.0)
    assert report["negative"]["lcb"] >= 0.97
    assert report["positive"]["lcb"] >= 0.94


def test_resolved_revision_from_runtime_manifest_is_preserved():
    truth = _truth()
    runtime = _runtime("Qwen/Qwen3.5-4B", "qwen35")
    runtime.pop("revision")
    runtime["resolved_revision"] = "f" * 40
    report = score_teacher(truth, _labels(truth), runtime)
    assert report["revision"] == "f" * 40


def test_teacher_over_vram_cap_fails_closed():
    truth = _truth()
    report = score_teacher(truth, _labels(truth), _runtime("too-big", "huge", vram=7.9))
    assert report["eligible"] is False
    assert "peak_vram" in report["failed_gates"]


def test_invalid_json_rate_disqualifies_teacher():
    truth = _truth()
    labels = _labels(truth, invalid=100)
    report = score_teacher(truth, labels, _runtime("bad-json", "bad-json"))
    assert report["json_valid_rate"] < 0.98
    assert report["eligible"] is False
    assert "json_valid_rate" in report["failed_gates"]


def test_same_family_pair_is_rejected_even_if_accurate():
    truth = _truth()
    labels = _labels(truth)
    first = score_teacher(truth, labels, _runtime("Qwen/A", "qwen35"))
    second = score_teacher(truth, labels, _runtime("Qwen/B", "qwen35"))
    pair = score_pair(truth, labels, labels, first, second)
    assert pair["eligible"] is False
    assert "independent_family" in pair["failed_gates"]


def test_pair_selection_prefers_consensus_quality_before_throughput():
    reports = {
        "qwen": {"eligible": True, "model_id": "Qwen/Qwen3.5-4B", "revision": "1" * 40, "family": "qwen35"},
        "gemma": {"eligible": True, "model_id": "google/gemma-4-E2B-it", "revision": "2" * 40, "family": "gemma4"},
        "euro": {"eligible": True, "model_id": "utter-project/EuroLLM-1.7B-Instruct", "revision": "3" * 40, "family": "eurollm"},
    }
    pairs = [
        {
            "teachers": ["qwen", "gemma"], "eligible": True,
            "consensus_precision": 0.999, "critical_precision": 0.998,
            "coverage": 0.72, "rows_per_second": 1.5,
        },
        {
            "teachers": ["qwen", "euro"], "eligible": True,
            "consensus_precision": 0.997, "critical_precision": 0.997,
            "coverage": 0.90, "rows_per_second": 8.0,
        },
    ]
    selected = select_teacher_pair(reports, pairs)
    assert selected["selected"] == ["qwen", "gemma"]


def test_no_eligible_pair_fails_closed():
    with pytest.raises(RuntimeError, match="no eligible teacher pair"):
        select_teacher_pair({}, [{"teachers": ["a", "b"], "eligible": False}])
